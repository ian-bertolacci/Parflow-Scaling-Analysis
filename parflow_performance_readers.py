# Types
import collections, enum

# Iterables
import itertools

# Data and analysis
import pandas as pd

# Parsing
import re, json, csv, fileinput

# Mathematical and Numerical
import math
from numpy import nan, inf

# Debug
from pprint import pprint

# Other
import os,random, copy

# My analysis framework
from analysis.utils import *
from analysis.parflow_performance_readers import *


from analysis.utils import *

def do_and_return_nothing_fn( *args, **kwargs ):
  pass

def load_output_directory_as_tree_structured_data( *outputs_directory_trees ):
  env_rx = re.compile( r"Environment:\n(.+)", flags=re.DOTALL )
  omp_rx = re.compile( r"^(?P<var>OMP_[\w_]+)=(?P<value>.+)$", flags=re.MULTILINE )
  case_name_rx = re.compile( r"^case" )
  topology_rx = re.compile( r"Test Configuration:\s+(?P<PX>\d+)\s+(?P<PY>\d+)\s+(?P<PZ>\d+)")
  timing_csv_filename_rx = re.compile( r".+\.out\.timing\.csv$" )
  elapsed_bin_time_rx = re.compile( r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ((?P<hours>\d+):)?(?P<minutes>\d+):(?P<seconds>\d+(\.\d+)?)" )

  all_data = []

  for output_directory in outputs_directory_trees:
    if not os.path.isdir( output_directory ):
      raise RuntimeError( f"{output_directory} is not a valid path to a directory" )

    dated_dirs = list( iterdirs( output_directory ) )

    if len(dated_dirs) < 1:
      raise RuntimeError( f"{output_directory} is has no run directories" )

    for dated_directory in dated_dirs:
      full_dated_directory = os.path.join( output_directory, dated_directory)
      case_dirs = list( iterdirs( full_dated_directory, case_name_rx ) )

      if len(case_dirs) < 1:
        raise RuntimeError( f"{full_dated_directory} has no case directories" )

      for case_directory in case_dirs:
        full_case_directory = os.path.join( full_dated_directory, case_directory )

        # Get case data
        case = dict(
          case=case_directory,
        )

        case_log = "".join( fileinput.input(f"{full_case_directory}/test_case.log") )

        env_info = env_rx.search( case_log ).group(1)

        omp_settings = { match.group("var") : match.group("value") for match in omp_rx.finditer( env_info ) }
        case["omp_settings"] = proper_type_dict( omp_settings )
        case["topology"] = { name : int(value) for (name,value) in topology_rx.search( case_log ).groupdict().items() }
        case["parallel_units"] = case["topology"]["PX"] * case["topology"]["PY"] * case["topology"]["PZ"] * ( case["omp_settings"]["OMP_NUM_THREADS"] if "OMP_NUM_THREADS" in case["omp_settings"] else 1 )

        case["trials"] = []
        for trial in iterdirs( f"{full_case_directory}/trials" ):
          full_trial_path=os.path.join( output_directory, dated_directory, case_directory, "trials", trial )
          timing_files = list( iterfiles( full_trial_path, timing_csv_filename_rx )  )
          if len(timing_files) > 1:
            raise RuntimeError( f"Multiple timing files in {full_trial_path}: {timing_files}" )
          elif len(timing_files) == 0:
            raise RuntimeError( f"No timing files in {full_trial_path}" )
          timing_file = timing_files[0]

          with open( os.path.join( full_trial_path, timing_file), "r" ) as csv_file:
            reader = csv.DictReader( csv_file )
            internal_timer_data = {
              row["Timer"] :
                proper_type_dict(
                  {
                    key: value
                     for (key,value) in row.items()
                  }
                )
              for row in reader
            }

          run_time_log = "".join( fileinput.input( os.path.join( full_trial_path, "run_time.log") ) )
          elapsed_match = elapsed_bin_time_rx.search( run_time_log )

          elapsed_seconds = (60*60*float(elapsed_match.group("hours")) if elapsed_match.group("hours") != None else 0) + 60*float(elapsed_match.group("minutes")) + float(elapsed_match.group("seconds"))
          external_timer_data = dict( elapsed = elapsed_seconds )
          # TODO: additional metrics from time, such as aximum resident set size of the process.

          case["trials"].append( dict( internal = internal_timer_data, external = external_timer_data ) )
        all_data.append( case )

  return all_data


def create_data_frame_from_tree_structured_data( data_list ):
  list_data = []
  for case in data_list:
    base_case_info = { "case" : case["case"], **case["omp_settings"], **case["topology"], "parallel_units" : case["parallel_units"] }
    for trial in case["trials"]:
      flat_internal_timings = {  f"{name} {key}" : value
                   for name, timer in trial["internal"].items()
                   for key, value in timer.items()
                   if key not in ["Timer", "MFLOPS (mops/s)", "FLOP (op)"] }
      full_trial_case_data = { **base_case_info, **flat_internal_timings, **trial["external"] }
      list_data.append( full_trial_case_data )
  data_framed = pd.DataFrame( list_data )
  return data_framed

def load_output_directory_as_data_frame( *outputs_directory_trees ):
  tree_data = load_output_directory_as_tree_structured_data( *outputs_directory_trees )
  data_framed = create_data_frame_from_tree_structured_data( tree_data )
  return data_framed


class ParflowScalingExperiment:
  default_default_key_values = dict(
    OMP_NUM_THREADS = 1
  )

  auto_name = lambda self: f"{self.domain} with {self.parflow_simple_name} ({self.parflow_long_version}) using {self.parallelism} on {self.system}"
  # default_notes = ""

  def __init__( self, parallelism, domain, system, parflow_simple_name, parflow_long_version, data, name = auto_name, default_key_values = default_default_key_values, clone_data = True ):
    self.parallelism = parallelism
    self.domain = domain
    self.system = system
    self.parflow_simple_name = parflow_simple_name
    self.parflow_long_version = parflow_long_version

    if clone_data:
      self.data = data.copy( deep = True )
    else:
      self.data = data

    # set unset keys to default value
    for key in set(default_key_values.keys()) - set(self.data.columns):
      self.data[key] = default_key_values[key]

    self.data = self.data.sort_values( ["PX","PY","PZ", "OMP_NUM_THREADS"] ).reset_index( drop=True )

    self.group_by_parallel_units_mean  = self.data.groupby( ["parallel_units"],                 as_index=False ).mean()
    self.group_by_parallel_units_min   = self.data.groupby( ["parallel_units"],                 as_index=False ).min()
    self.group_by_parallel_decomp_mean = self.data.groupby( ["PX","PY","PZ","OMP_NUM_THREADS"], as_index=False ).mean()
    self.group_by_parallel_decomp_min  = self.data.groupby( ["PX","PY","PZ","OMP_NUM_THREADS"], as_index=False ).min()

    self.parallel_units = sorted( get_actual_set( self.data, "parallel_units" ) )

    if callable(name):
      self.name = name(self)
    else:
      self.name = name

  # predicate_fn: dataframe -> bool (evaluates an inner indexing expression. Common example: lambda df: df.parallel_units == 1 )
  # true_update_fn:  ParflowScalingExperiment -> none (impertively apply updates on ParflowScalingExperiment resulting from predicate evaluating to true )
  # false_update_fn: ParflowScalingExperiment -> none (impertively apply updates on ParflowScalingExperiment resulting from predicate evaluating to false )
  # update_fn:       ParflowScalingExperiment -> none (impertively apply updates on both ParflowScalingExperiments regardless of how they evaluate on the predicate )
  def split_into_experiments_by_predicate_on_data( self, predicate_fn, true_update_fn = do_and_return_nothing_fn, false_update_fn = do_and_return_nothing_fn, update_fn = do_and_return_nothing_fn ):
    temp_pred_key = f"__pred_split__{self.name}-{self.system}-{self.domain}-{self.parflow_simple_name}-{self.parflow_long_version}"
    self.data[temp_pred_key] = predicate_fn(self.data)
    true_data  = self.data[ self.data[temp_pred_key] == True  ].drop( columns=temp_pred_key ).reset_index( drop = True )
    false_data = self.data[ self.data[temp_pred_key] == False ].drop( columns=temp_pred_key ).reset_index( drop = True )

    self.data[temp_pred_key].drop( columns=temp_pred_key )

    true_data_class = ParflowScalingExperiment(
      name = self.name,
      parallelism = self.parallelism,
      domain = self.domain,
      system = self.system,
      parflow_simple_name = self.parflow_simple_name,
      parflow_long_version = self.parflow_long_version,
      data = true_data
    )

    false_data_class = ParflowScalingExperiment(
      name = self.name,
      parallelism = self.parallelism,
      domain = self.domain,
      system = self.system,
      parflow_simple_name = self.parflow_simple_name,
      parflow_long_version = self.parflow_long_version,
      data = false_data
    )

    update_fn( true_data_class )
    update_fn( false_data_class )

    true_update_fn( true_data_class )
    false_update_fn( false_data_class )

    return true_data_class, false_data_class

  def filter_if( self, predicate_fn, update_fn = do_and_return_nothing_fn ):
    filtered_out, filter_keep = self.split_into_experiments_by_predicate_on_data( predicate_fn = predicate_fn, false_update_fn = update_fn )
    return filter_keep

  # update_fn: (experiment, group_tuple) -> void
  def split_into_experiments_by_parallelism( self, update_fn = None, include_serial=False, rename_fn = auto_name ):
    def set_parallelism( experiment ):
      omp_num_threads = get_actual_set( experiment.data, "OMP_NUM_THREADS" )
      mpi_processes = [ px*py*pz for px, py, pz in get_actual_set( experiment.data, "PX", "PY", "PZ" ) ]

      used_MPI = False
      used_OMP = False

      if omp_num_threads != set([1]):
        used_OMP = True
      if max( mpi_processes ) > 1:
        used_MPI = True

      if include_serial:
        if used_MPI and used_OMP:
          parallelism = "MPI+OpenMP"
        elif used_MPI:
          parallelism = "MPI"
        elif used_OMP:
          parallelism = "OpenMP"
        else:
          parallelism = "Serial"
      else:
        parallelism = "MPI"
        if used_OMP:
          parallelism += "+OpenMP"

      experiment.parallelism = parallelism
      experiment.name = rename_fn( experiment )

    return self.split_into_experiments_by_predicate_on_data( lambda df: df.OMP_NUM_THREADS == 1, update_fn = set_parallelism )


  # update_fn: (experiment, group_tuple) -> void
  def split_into_experiments_by_grouped_key_values( self, grouping_keys, update_fn = None ):
    if update_fn == None:
      def update_fn(experiment, group_tuple):
        experiment.name = f"{experiment.name} ({', '.join( (str(i) for i in group_tuple) )})"

    experiments = []
    for (group, group_data) in self.data.groupby( grouping_keys, as_index=False ):
      if not isinstance(group, tuple):
        group = (group,)
      new_experiment = self.clone_with_other_data( group_data )
      update_fn( new_experiment, group )
      experiments.append( new_experiment )
    return experiments

  #   update_fn: (experiment, group_tuple) -> void
  def split_into_experiments_by_topology( self, additional_grouping_keys = [], update_fn = None ):
    topology_keys = ["PX", "PY", "PZ"]
    grouping_keys = list( topology_keys ) # copy because why not
    # Add keys not previously used, while maintaining ordering of keys
    for key in additional_grouping_keys:
      if key not in grouping_keys:
        grouping_keys.append( key )

    return split_into_experiments_by_grouped_key_values( grouping_keys, update_fn )

  # update_fn: (experiment, group_tuple) -> void
  def split_into_group_by_grouped_key_value( self, grouping_keys, group_name = None, update_fn = None ):
    if group_name == None:
      group_name = f"{self.name} grouped {tuple(grouping_keys)}"

    if update_fn == None:
      def update_fn( experiment, group_tuple ):
        experiment.name = f"{experiment.name} {tuple(group_tuple)}"

    experiments = self.split_into_experiments_by_grouped_key_values( grouping_keys, update_fn )
    return DataGroup( group_name, experiments )

  def split_into_group_by_topology( self, additional_grouping_keys=[], group_name = None, update_fn = None ):
    topology_keys = ["PX", "PY", "PZ"]
    grouping_keys = list( topology_keys )
    # Add keys not previously used, while maintaining ordering of keys
    for key in additional_grouping_keys:
      if key not in grouping_keys:
        grouping_keys.append( key )

    if group_name == None:
      name = f"{self.name} grouped {tuple(grouping_keys)}"

    experiments = self.split_into_experiments_by_topology( grouping_keys, update_fn )
    return DataGroup( group_name, experiments )

  def clone(self, clone_data=True):
    return self.clone_with_other_data( self.data, clone_data )

  def clone_with_other_data(self, data, clone_data=False):
    experiment = ParflowScalingExperiment(
      name = self.name,
      parallelism = self.parallelism,
      domain = self.domain,
      system = self.system,
      parflow_simple_name = self.parflow_simple_name,
      parflow_long_version = self.parflow_long_version,
      data = data,
      clone_data = clone_data
    )
    return experiment

  def create_mock(self):
    mock_ParflowScalingExperiment = collections.namedtuple( "mock_ParflowScalingExperiment", ["name", "parallelism", "domain", "system", "parflow_simple_name", "parflow_long_version"] )
    return mock_ParflowScalingExperiment( **{ attr : getattr(self,attr) for attr in mock_ParflowScalingExperiment._fields } )

  def __str__(self):
    if self.name == ParflowScalingExperiment.auto_name( self ):
      return f"Experiment {self.name}"
    else:
      return f"Experiment {self.name}, {ParflowScalingExperiment.auto_name( self )}"

  def __repr__(self):
    return str(self)


class ParflowScalingExperimentFromFiles(ParflowScalingExperiment):
  def __init__( self, files, *args, **kwargs ):
    self.files = files
    # Load data from files
    data = load_output_directory_as_data_frame( *self.files )
    super().__init__( *args, **kwargs, data=data )


class DataGroup:
  def __init__(self, name, experiments, clone=False):
    self.name = name
    self.experiments = [ e.clone() if clone else e for e in experiments ]
    self.all_data = pd.concat( [ data_class.data for data_class in self.experiments ] )

    self.all_data = self.all_data.sort_values(["PX","PY","PZ", "OMP_NUM_THREADS"]).reset_index(drop=True)

    numeric_columns = [ c for c in self.all_data.columns if self.all_data.dtypes[c].kind in 'biufc' ]

    # self.group_by_parallel_units_mean  = self.all_data[ numeric_columns ].groupby( ["parallel_units"],                 as_index=False ).mean()
    # self.group_by_parallel_units_min   = self.all_data[ numeric_columns ].groupby( ["parallel_units"],                 as_index=False ).min()
    # self.group_by_parallel_decomp_mean = self.all_data[ numeric_columns ].groupby( ["PX","PY","PZ","OMP_NUM_THREADS"], as_index=False ).mean()
    # self.group_by_parallel_decomp_min  = self.all_data[ numeric_columns ].groupby( ["PX","PY","PZ","OMP_NUM_THREADS"], as_index=False ).min()

    self.parallel_units = sorted( get_actual_set( self.all_data, "parallel_units" ) )

  def clone(self, new_name=None, clone_experiments=True):
    if new_name == None:
      new_name = self.name
    return DataGroup( new_name, self.experiments, clone=True )

  # group_fn: ParflowScalingExperiment -> hashable
  # name_fn: DataGroup, mock_ParflowScalingExperiment, list_of_experiments_in_group -> str
  def split_into_groups_by_function(self, group_fn, name_fn = None ):
    if name_fn == None:
      name_fn = lambda self, mock, key, group_experiments: f"{self.name} ({key})"

    group_dict = dict()
    group_mock = dict()
    for experiment in self.experiments:
      group_key = group_fn( experiment )
      if group_key not in group_dict:
        group_dict[group_key] = []
        group_mock[group_key] = experiment.create_mock()
      group_dict[group_key].append( experiment )

    for key in group_dict.keys():
      group_experiments = group_dict[key]
      group_dict[key] = DataGroup( name_fn(self, group_mock[key], key, group_experiments), group_experiments )

    return DataGroupList( group_dict.values() )

  def split_into_groups_by_parallelism(self, name_fn = None):
    return self.split_into_groups_by_function( lambda e: e.parallelism, name_fn )

  def split_into_groups_by_domain(self, name_fn = None):
    return self.split_into_groups_by_function( lambda e: e.domain, name_fn )

  def split_into_groups_by_system(self, name_fn = None):
    return self.split_into_groups_by_function( lambda e: e.system, name_fn )

  def split_into_groups_by_parflow_simple_name(self, name_fn = None):
    return self.split_into_groups_by_function( lambda e: e.parflow_simple_name, name_fn )

  def split_into_groups_by_parflow_long_version(self, name_fn = None):
    return self.split_into_groups_by_function( lambda e: e.parflow_long_version, name_fn )

  def split_into_groups_by_name(self, name_fn = None):
    return self.split_into_groups_by_function( lambda e: e.name, name_fn )

  def apply_update(self, fn, inplace=False, modifying_fn=True, clone=True, new_name=None):
    if new_name == None:
      new_name = self.name

    def wrapped_fn( experiment ):
      if clone:
        experiment = experiment.clone()
      if modifying_fn:
        fn( experiment )
        return_value = experiment
      else:
        return_value = fn( experiment )
      return return_value

    experiments = [ wrapped_fn( e ) for e in self.experiments ]

    if inplace:
      self.name = new_name
      self.experiments = experiments
      return_value = self
    else:
      data_group = DataGroup( new_name, experiments )
      return_value = data_group
    return return_value

  def apply_update_inplace(self, fn, modifying_fn=True, clone=False, new_name=None):
    return self.apply_update(fn, inplace=True, modifying_fn=modifying_fn, clone=clone, new_name=new_name)

  # fn: ParflowScalingExperiment -> value (should not modify input)
  def map(self, fn):
    return map( fn, self.experiments )

  # fn: state, ParflowScalingExperiment -> state' (should not modify input)
  def reduce(self, fn, initial):
    return functools.reduce( fn, self.experiments, initial )

  def collect_key_values(self, *keys):
    def reduce_fn( value, experiment ):
      return value | get_actual_set( experiment.data, *keys )
    return self.reduce( reduce_fn, set() )

  def collect_attribute_values(self, *attributes):
    def reduce_fn( value, experiment ):
      return value | set( ( getattr(experiment, attr) for attr in attributes ) )
    return self.reduce( reduce_fn, set() )


  def __getitem__(self, experiment_names):
    if not (isinstance(experiment_names,list) or isinstance(experiment_names,tuple) or isinstance(experiment_names,set)):
      experiment_names = [ experiment_names ]
    classes = []
    for experiment_name in experiment_names:
      experiment = None
      for experiment_iter in self.experiments:
        if experiment_iter.name == experiment_name:
          experiment = experiment_iter
          break

      if experiment == None:
        raise KeyError( f"Experiment named {experiment_name} not in group." )
      classes.append( data_class )

    if len(classes) == 0:
      return_value = None
    elif len(classes) == 1:
      return_value = classes[0]
    else:
      return_value = classes
    return return_value

  def __str__(self):
    ordered_keys = [ "parallelism", "domain", "system", "parflow_simple_name", "parflow_long_version", "name" ]
    max_width_keys = { key : len(key) for key in ordered_keys }

    for e in self.experiments:
      for attribute in max_width_keys.keys():
        max_width_keys[attribute] = max( max_width_keys[attribute], len(getattr(e, attribute)) )

    row_format_string = "    ".join(
      itertools.starmap(
        lambda i, key : f"{{{i}:<{max_width_keys[key]}}}",
        enumerate( ordered_keys )
      )
    )

    ret_string = f"Group \"{self.name}\" containing {len(self.experiments)} experiments:\n\t" + \
    row_format_string.format( *ordered_keys ) + "\n\t" + \
      "\n\t".join(
        (
          row_format_string.format( *( getattr(e, attribute) for attribute in ordered_keys ) )
          for e in self.experiments
        )
      )
    return ret_string

  # def __repr__(self):
    # return f"Group \"{self.name}\" containing {len(self.experiments)} experiments:\n\t" + "\n\t".join( (str(e) for e in self.experiments) )
    # return str( self )

  def join_these(name, *others):
    data_group = DataGroup(
      name = name,
      experiments = [ g.experiments for g in others ]
    )
    return data_group

  def join(self, name, *others):
    return DataGroup.join_these(name, self, *others)

  def __iter__(self):
    return iter(self.experiments)


class DataGroupList:
  def __init__(self, groups):
    self.groups = list( recursive_unpack(groups, do_not_unpack_when_true=lambda x: isinstance(x,DataGroup)) )

  def apply_function_to_groups(self, function, *args, **kwargs):
    l = list( ( function(g, *args, **kwargs ) for g in self.groups ) )
    return DataGroupList( l )

  def split_into_groups_by_function(self, *args, **kwargs ):
    return self.apply_function_to_groups( DataGroup.split_into_groups_by_function, *args, **kwarg)

  def split_into_groups_by_parallelism(self, *args, **kwargs):
    return self.apply_function_to_groups( DataGroup.split_into_groups_by_parallelism, *args, **kwargs )

  def split_into_groups_by_domain(self, *args, **kwargs ):
    return self.apply_function_to_groups( DataGroup.split_into_groups_by_domain, *args, **kwargs )

  def split_into_groups_by_system(self, *args, **kwargs ):
    return self.apply_function_to_groups( DataGroup.split_into_groups_by_system, *args, **kwargs )

  def split_into_groups_by_parflow_simple_name(self, *args, **kwargs ):
    return self.apply_function_to_groups( DataGroup.split_into_groups_by_parflow_simple_name, *args, **kwargs )

  def split_into_groups_by_parflow_long_version(self, *args, **kwargs ):
    return self.apply_function_to_groups( DataGroup.split_into_groups_by_parflow_long_version, *args, **kwargs )

  def split_into_groups_by_name(self, *args, **kwargs ):
    return self.apply_function_to_groups( DataGroup.split_into_groups_by_name, *args, **kwargs )

  def apply_update(self, *args, **kwargs ):
    return self.apply_function_to_groups( DataGroup.apply_update, *args, **kwargs )

  def apply_update_inplace(self, *args, **kwargs):
    return self.apply_function_to_groups( DataGroup.apply_update_inplace, *args, **kwargs )

  def apply_group_update(self, fn, modifying_fn=True, clone=False):
    def wrapped_fn( group ):
      if clone:
        group = group.clone()

      if modifying_fn:
        fn( group )
        return_value = group
      else:
        return_value = fn( group )
      return return_value

  # fn: ParflowScalingExperiment -> value (should not modify input)
  def map(self, *args, **kwargs):
    return itertools.chain( *( g.map( *args, **kwargs ) for g in self.groups) )

  # fn: state, ParflowScalingExperiment -> state' (should not modify input)
  def reduce(self, fn, initial):
    return functools.reduce( fn, itertools.chain( *( g.experiments for g in self.groups ) ), initial )

  def collect_key_values(self, *keys):
    def reduce_fn( value, experiment ):
      return value | get_actual_set( experiment.data, *keys )
    return self.reduce( reduce_fn, set() )

  def collect_attribute_values(self, *attributes):
    def reduce_fn( value, experiment ):
      return value | set( ( getattr(experiment, attr) for attr in attributes ) )
    return self.reduce( reduce_fn, set() )

  def join_groups(self, name):
    return DataGroup.join_these_groups(name, *self.groups)

  def concatenate_these( *data_group_lists ):
    return DataGroupList( itertools.chain( *( l.groups for l in data_group_lists) ) )

  def concatenate(self, *others):
    return DataGroupList.concatenate_these( self, *others )

  def __iter__(self):
    return iter(self.groups)

  def __str__(self):
    return "\n".join( (str(g) for g in self.groups) )
