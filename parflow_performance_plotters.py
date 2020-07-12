import os, itertools, collections, re, json, csv, math, scipy, scipy, scipy.stats, plotly, fileinput, random, copy
import numpy as np
import pandas as pd
import sympy as sp

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

from numpy import nan, inf
from pprint import pprint

from analysis.utils import *
from analysis.parflow_performance_readers import *

class ParflowScalingExperimentPlotter:
  default_metric_key = "Total Runtime Time (s)"
  default_metric_units = "seconds"
  default_metric_transformation_fn = lambda x: x
  default_min_parallel_units = -inf
  default_max_parallel_units = inf

  def __init__(self, experiment : ParflowScalingExperiment, color, symbol, metric_key = default_metric_key, metric_units = default_metric_units, metric_transformation_fn = default_metric_transformation_fn, min_parallel_units = default_min_parallel_units, max_parallel_units = default_max_parallel_units):
    self.experiment = experiment
    self.color = color
    self.symbol = symbol
    self.metric_key = metric_key
    self.metric_units = metric_units
    self.metric_transformation_fn = metric_transformation_fn
    self.min_parallel_units = min_parallel_units
    self.max_parallel_units = max_parallel_units

  def make_plots( self, symbol_size = 10, show_scatter_in_legend = True, show_line_in_legend = False ):
    scatter_plot      = self.make_scatter_plot(      show_in_legend=show_scatter_in_legend, symbol_size=symbol_size,  )
    average_line_plot = self.make_average_line_plot( show_in_legend=show_line_in_legend )
    minium_line_plot  = self.make_min_line_plot(     show_in_legend=show_line_in_legend )
    return scatter_plot, average_line_plot, minium_line_plot

  def make_scatter_plot( self, symbol_size = 10, show_in_legend = True ):
    scoped_data = self.experiment.data[ (self.min_parallel_units <= self.experiment.data["parallel_units"]) & ( self.experiment.data["parallel_units"] <= self.max_parallel_units) ]
    scatter_plot = go.Scatter(
      x=scoped_data["parallel_units"],
      y=self.metric_transformation_fn(scoped_data[self.metric_key] ),
      name=self.experiment.name,
      mode="markers",
      marker=dict(
        color=self.color,
        symbol=self.symbol,
        size=symbol_size
      ),
      showlegend=show_in_legend
    )

    return scatter_plot

  def make_average_line_plot( self, symbol_size = 10, show_in_legend = True ):
    scoped_data = self.experiment.group_by_parallel_units_mean[ (self.min_parallel_units <=  self.experiment.group_by_parallel_units_mean["parallel_units"]) & ( self.experiment.group_by_parallel_units_mean["parallel_units"] <= self.max_parallel_units) ]
    average_line_plot = go.Scatter(
      x=scoped_data["parallel_units"],
      y=self.metric_transformation_fn( scoped_data[self.metric_key] ),
      name=f"{self.experiment.name} (averaged)",
      mode="lines",
      line=dict( color=self.color ),
      showlegend=show_in_legend,
    )
    return average_line_plot

  def make_min_line_plot( self, symbol_size = 10, show_in_legend = True ):
    scoped_data = self.experiment.group_by_parallel_units_min[ (self.min_parallel_units <=  self.experiment.group_by_parallel_units_mean["parallel_units"]) & ( self.experiment.group_by_parallel_units_mean["parallel_units"] <= self.max_parallel_units) ]
    minimum_line_plot = go.Scatter(
      x=scoped_data["parallel_units"],
      y=self.metric_transformation_fn( scoped_data[self.metric_key] ),
      name=f"{self.experiment.name} (minimum)",
      mode="lines",
      line=dict( color=self.color, dash='dash' ),
      showlegend=show_in_legend,
    )

    return minimum_line_plot

  def groups_ParflowScalingExperimentPlotter( self, aggregating_keys = ["PX", "PY", "PZ", "OMP_NUM_THREADS"], marker_factory=None ):
    for experiment in self.experiment.group_into_ParflowScalingExperimentes( aggregating_keys ):
      if marker_factory == None:
        color = self.color
        symbol = self.symbol
      else:
        marker = marker_factory.take()
        color = marker[0]
        symbol = marker[1]

      yield ParflowScalingExperimentPlotter( experiment, color, symbol, self.metric_key, self.metric_units, self.metric_transformation_fn, self.min_parallel_units, self.max_parallel_units )

  def group_into_DataGroupPlotter( self, aggregating_keys = ["PX", "PY", "PZ", "OMP_NUM_THREADS"], marker_factory=None ):
    data_group = self.experiment.group_into_DataGroup( aggregating_keys )
    if marker_factory == None:
      marker_factory = RotatingFactory( [(self.color, self.symbol)] )
    data_group_plotter = DataGroupPlotter(
      data_group=data_group,
      marker_factory=marker_factory,
      metric_key = self.metric_key,
      metric_name = None,
      metric_units = self.metric_units,
      metric_transformation_fn = self.metric_transformation_fn,
      min_parallel_units = self.min_parallel_units,
      max_parallel_units = self.max_parallel_units,
      aggregated_groups = [],
      aggregating_keys = []
    )
    return data_group_plotter

  def make_aggregated_plots( self, aggregating_keys = ["PX", "PY", "PZ", "OMP_NUM_THREADS"], marker_factory=None, symbol_size = 10, show_scatter_in_legend = True, show_line_in_legend = False ):
    return  self.group_into_DataGroupPlotter(
              aggregating_keys=aggregating_keys,
              marker_factory=marker_factory
            ).make_classes_plots(
              symbol_size=symbol_size,
              show_scatter_in_legend=show_scatter_in_legend,
              show_line_in_legend=show_line_in_legend
            )


  def make_aggregated_scatter_plots( self, aggregating_keys = ["PX", "PY", "PZ", "OMP_NUM_THREADS"], marker_factory=None, symbol_size = 10, show_in_legend = True ):
    return  self.group_into_DataGroupPlotter(
              aggregating_keys=aggregating_keys,
              marker_factory=marker_factory
            ).make_classes_scatter_plots(
              symbol_size=symbol_size,
              show_in_legend=show_in_legend
            )

  def make_aggregated_average_line_plots( self, aggregating_keys = ["PX", "PY", "PZ", "OMP_NUM_THREADS"], marker_factory=None, symbol_size = 10, show_in_legend = False ):
    return  self.group_into_DataGroupPlotter(
              aggregating_keys=aggregating_keys,
              marker_factory=marker_factory
            ).make_classes_average_line_plots(
              symbol_size=symbol_size,
              show_in_legend=show_in_legend
            )

  def make_aggregated_average_line_plots( self, aggregating_keys = ["PX", "PY", "PZ", "OMP_NUM_THREADS"], marker_factory=None, symbol_size = 10, show_in_legend = False ):
    return  self.group_into_DataGroupPlotter(
              aggregating_keys=aggregating_keys,
              marker_factory=marker_factory
            ).make_classes_average_line_plots(
              symbol_size=symbol_size,
              show_in_legend=show_in_legend
            )


class DataGroupPlotter:
  def __init__(self, data_group, marker_factory, metric_key = ParflowScalingExperimentPlotter.default_metric_key, metric_name = None, metric_units = ParflowScalingExperimentPlotter.default_metric_units, metric_transformation_fn = ParflowScalingExperimentPlotter.default_metric_transformation_fn, min_parallel_units = ParflowScalingExperimentPlotter.default_min_parallel_units, max_parallel_units = ParflowScalingExperimentPlotter.default_max_parallel_units, aggregated_groups = [], aggregating_keys = [] ):
    self.data_group = data_group
    self.marker_factory = copy.deepcopy(marker_factory)
    self.metric_key = metric_key
    self.metric_name = metric_name if metric_name != None else self.metric_key
    self.metric_units = metric_units
    self.metric_transformation_fn = metric_transformation_fn
    self.min_parallel_units = min_parallel_units
    self.max_parallel_units = max_parallel_units
    self.aggregated_groups = aggregated_groups
    self.aggregating_keys = aggregating_keys

    self.previously_used_markers = {
      key : self.marker_factory.take()
      for key in ["serial"]
    }

    self.plotters = []
    for experiment in self.data_group.experiments:
      marker = self.marker_factory.take()
      self.plotters.append( ParflowScalingExperimentPlotter( experiment, marker[0], marker[1], metric_key = self.metric_key, metric_units = self.metric_units, metric_transformation_fn = self.metric_transformation_fn, min_parallel_units = self.min_parallel_units, max_parallel_units = self.max_parallel_units )  )

    self.all_serial = self.data_group.all_data[ self.data_group.all_data["parallel_units"] == 1 ]
    self.serial_mean = self.all_serial[self.metric_key].mean()


  def make_serial_plots(self, name=f"Serial", symbol_size=10, show_scatter_in_legend = True, show_line_in_legend = False ):
    serial_scatter_plot = self.make_serial_scatter_plot( name=name, symbol_size=symbol_size, show_in_legend=show_scatter_in_legend )
    serial_average_line_plot = self.make_serial_average_line_plot( name=name, symbol_size=symbol_size, show_in_legend=show_line_in_legend )
    return serial_scatter_plot, serial_average_line_plot

  def make_serial_scatter_plot(self, name=f"Serial", symbol_size=10, show_in_legend=False ):
    if "serial" not in self.previously_used_markers:
      self.previously_used_markers["serial"] = self.marker_factory.take()

    serial_scatter_plot = go.Scatter(
      x=self.all_serial["parallel_units"],
      y=self.metric_transformation_fn( self.all_serial[self.metric_key] ),
      name=name,
      mode="markers",
      marker=dict(
        color=self.previously_used_markers["serial"][0],
        symbol=self.previously_used_markers["serial"][1],
        size=symbol_size
      ),
      showlegend=show_in_legend
    )

    return serial_scatter_plot

  def make_serial_average_line_plot(self, name=f"Serial", symbol_size=10, show_in_legend=True, force_name=None ):
    if "serial" not in self.previously_used_markers:
      self.previously_used_markers["serial"] = self.marker_factory.take()

    using_name = f"{name} (averaged)" if force_name == None else force_name

    scoped_parallel_units = [ parallel_units for parallel_units in self.data_group.parallel_units if self.min_parallel_units <= parallel_units and parallel_units <= self.max_parallel_units ]
    serial_line_plot = go.Scatter(
      x=scoped_parallel_units,
      y=[ self.metric_transformation_fn( self.serial_mean ) ]*len(scoped_parallel_units),
      name=using_name,
      mode="lines",
      line=dict( color=self.previously_used_markers["serial"][0] ),
      showlegend=show_in_legend
    )

    return serial_line_plot

  def make_classes_plots( self, show_scatter_in_legend = True, show_line_in_legend = False, *args, **kwargs ):
    plots = list(
              itertools.chain(
                *zip(
                  self.make_classes_scatter_plots(      *args, **kwargs, show_in_legend=show_scatter_in_legend ),
                  self.make_classes_average_line_plots( *args, **kwargs, show_in_legend=show_line_in_legend ),
                  self.make_classes_min_line_plots(     *args, **kwargs, show_in_legend=show_line_in_legend )
                )
              )
            )
    return plots

  def make_classes_scatter_plots( self, *args, **kwargs ):
    return [ plotter.make_scatter_plot( *args, **kwargs ) for plotter in self.plotters ]

  def make_classes_average_line_plots( self, *args, **kwargs ):
    return [ plotter.make_average_line_plot( *args, **kwargs ) for plotter in self.plotters ]

  def make_classes_min_line_plots( self, *args, **kwargs ):
    return [ plotter.make_min_line_plot( *args, **kwargs ) for plotter in self.plotters ]


  # functions takes: plotter, returns called function
  def flat_apply_function_across_aggregated( self, grouped_function, normal_function, extra_aggregated_groups=[], extra_aggregating_keys=[], *args, **kwargs ):
    aggregated_groups = self.get_aggregated_groups( extra_aggregated_groups )
    aggregating_keys = self.get_aggregated_keys( extra_aggregating_keys )
    plots = []
    for plotter in self.plotters:
      if plotter.experiment.name in aggregated_groups:
        value = grouped_function(plotter)( *args, **kwargs, aggregating_keys=aggregating_keys, marker_factory=self.marker_factory.clone() )
      else:
        value = normal_function(plotter)( *args, **kwargs )

      if isinstance( value, list ) or isinstance( value, tuple ):
        plots.extend( value )
      else:
        plots.append( value )

    return plots


  def make_classes_aggregated_plots( self, extra_aggregated_groups=[], extra_aggregating_keys=[], *args, **kwargs ):
    grouped_function = lambda plotter: plotter.make_aggregated_plots
    normal_function = lambda plotter: plotter.make_plots
    return self.flat_apply_function_across_aggregated( grouped_function, normal_function, extra_aggregated_groups, extra_aggregating_keys, *args, **kwargs )

  def make_classes_aggregated_scatter_plots( self, extra_aggregated_groups=[], extra_aggregating_keys=[], *args, **kwargs ):
    grouped_function = lambda plotter: plotter.make_aggregated_scatter_plots
    normal_function = lambda plotter: plotter.make_scatter_plot
    return self.flat_apply_function_across_aggregated( grouped_function, normal_function, extra_aggregated_groups, extra_aggregating_keys, *args, **kwargs )

  def make_classes_aggregated_average_line_plots( self, extra_aggregated_groups=[], extra_aggregating_keys=[], *args, **kwargs ):
    grouped_function = lambda plotter: plotter.make_aggregated_average_line_plots
    normal_function = lambda plotter: plotter.make_average_line_plot
    return self.flat_apply_function_across_aggregated( grouped_function, normal_function, extra_aggregated_groups, extra_aggregating_keys, *args, **kwargs )

  def make_classes_aggregated_min_line_plots( self, extra_aggregated_groups=[], extra_aggregating_keys=[], *args, **kwargs ):
    grouped_function = lambda plotter: plotter.make_aggregated_min_line_plots
    normal_function = lambda plotter: plotter.make_min_line_plot
    return self.flat_apply_function_across_aggregated( grouped_function, normal_function, extra_aggregated_groups, extra_aggregating_keys, *args, **kwargs )


  def make_default_title(self):
    versions = set( ( str(experiment.parflow_simple_name).strip() for experiment in self.data_group.experiments ) )
    multiple_versions = len(versions) > 1

    if multiple_versions:
      class_format_string = "{0}@{1}"
    else:
      class_format_string = "{0}"


    title_string = f"{self.data_group.name}<br>" + " vs ".join( ( class_format_string.format(experiment.name, experiment.parflow_simple_name) for experiment in self.data_group.experiments ) )
    # title_string = f"{self.data_group.name}<br>" + r"<table><tr><td>Comparing:</td><td>" + ("<br>".join( ( class_format_string.format(experiment.name, experiment.parflow_simple_name) for experiment in self.data_group.experiments ) ) ) + r"</td></tr></table>"
    if not multiple_versions:
      title_string += f" ({versions.pop()})"

    return title_string

  def make_figure( self, plots, title=None, subtitle=None, show=True, yaxis_scale="linear", symbol_size=10, plot_line=True, plot_scatter=True ):
    if title == None:
      title = self.make_default_title()
    if subtitle != None:
      title += "<br>" + subtitle

    fig = go.Figure(
      data=plots,
      layout=dict(
        title=title,
        xaxis=dict(
          title="Parallel units (PX * PY * PZ * OpenMP Threads)",
          tickmode="array",
          tickvals=self.data_group.parallel_units
        ),
        yaxis=dict(
          title=f"{self.metric_name} ({self.metric_units})",
          rangemode="tozero",
          type=yaxis_scale
        )
      )
    )

    if show:
      fig.show()

    return fig

  def plot_individual( self, title=None, subtitle=None, show=True, yaxis_scale="linear", symbol_size=10, plot_line=True, show_line_in_legend=False, plot_scatter=True, show_scatter_in_legend=True ):
    plots = []
    if plot_line and plot_scatter :
      plots = [
        *self.make_serial_plots(   symbol_size=symbol_size, show_scatter_in_legend=False, show_line_in_legend=True ),
        *self.make_classes_plots( symbol_size=symbol_size, show_line_in_legend=show_line_in_legend )
      ]
    elif plot_line:
      plots = [
        self.make_serial_average_line_plot(    symbol_size=symbol_size, show_in_legend=True ),
        *self.make_classes_average_line_plots( symbol_size=symbol_size, show_in_legend=True ),
        *self.make_classes_min_line_plots(     symbol_size=symbol_size, show_in_legend=True )
      ]
    elif plot_scatter:
      plots = [
        self.make_serial_scatter_plot(    symbol_size=symbol_size, show_in_legend=True ),
        *self.make_classes_scatter_plots( symbol_size=symbol_size, show_in_legend=True )
      ]
    figure = self.make_figure( plots, title=title, subtitle=subtitle, show=show, yaxis_scale=yaxis_scale, symbol_size=symbol_size, plot_line=plot_line, plot_scatter=plot_scatter )
    return figure

  def plot_aggregated( self, extra_aggregated_groups=[], extra_aggregating_keys=[], title=None, subtitle=None, show=True, yaxis_scale="linear", symbol_size=10, plot_line=True, show_line_in_legend=False, plot_scatter=True, show_scatter_in_legend=True ):
    plots = []

    # For string purposes only
    all_aggregated_groups = self.get_aggregated_groups( extra_aggregated_groups )
    all_aggregating_keys = self.get_aggregated_keys( extra_aggregating_keys )
    if len(all_aggregated_groups) > 0 and len(all_aggregating_keys) > 0:
      if subtitle == None:
        subtitle = ""
      else:
        subtitle += "<br>"

      if len(all_aggregated_groups) <= 2:
        subtitle += " and ".join( all_aggregated_groups )
      else: # len(all_aggregated_groups) > 2:
        subtitle +=  f"{', '.join( all_aggregated_groups[:-1])}, and {all_aggregated_groups[-1]}"
      subtitle += f" grouped by ({', '.join(all_aggregating_keys)})."

    if plot_line and plot_scatter:
      plots = [
        *self.make_serial_plots( symbol_size=symbol_size, show_scatter_in_legend=True, show_line_in_legend=True ),
        *self.make_classes_aggregated_plots( extra_aggregated_groups=extra_aggregated_groups, extra_aggregating_keys=extra_aggregating_keys, symbol_size=symbol_size, show_line_in_legend=show_line_in_legend )
      ]
    elif plot_line:
      plots = [
        self.make_serial_average_line_plot( symbol_size=symbol_size, show_in_legend=True ),
        *self.make_classes_aggregated_average_line_plots( extra_aggregated_groups=extra_aggregated_groups, extra_aggregating_keys=extra_aggregating_keys, symbol_size=symbol_size, show_in_legend=True )
        *self.make_classes_aggregated_min_line_plots( extra_aggregated_groups=extra_aggregated_groups, extra_aggregating_keys=extra_aggregating_keys, symbol_size=symbol_size, show_in_legend=True )
      ]
    elif plot_scatter:
      plots = [
        self.make_serial_scatter_plot( symbol_size=symbol_size, show_in_legend=True ),
        *self.make_classes_aggregated_scatter_plots( extra_aggregated_groups=extra_aggregated_groups, extra_aggregating_keys=extra_aggregating_keys, symbol_size=symbol_size, show_in_legend=True )
      ]

    return self.make_figure( plots, title=title, subtitle=subtitle, show=show, yaxis_scale=yaxis_scale, symbol_size=symbol_size, plot_line=plot_line, plot_scatter=plot_scatter )

  def plots( self, extra_aggregated_groups=[], extra_aggregating_keys=[], title=None, subtitle=None, show=True, yaxis_scale="linear", symbol_size=10, plot_line=True, plot_scatter=True ):
    individual = self.plot_individual( title=title, subtitle=subtitle, show=show, yaxis_scale=yaxis_scale, symbol_size=symbol_size, plot_line=plot_line, plot_scatter=plot_scatter )
    if len(self.get_aggregated_groups(extra_aggregated_groups)) > 0:
      grouped = self.plot_aggregated( extra_aggregated_groups=extra_aggregated_groups, extra_aggregating_keys=extra_aggregating_keys, title=title, subtitle=subtitle, show=show, yaxis_scale=yaxis_scale, symbol_size=symbol_size, plot_line=plot_line, plot_scatter=plot_scatter )
    else:
      grouped = None
    return individual, grouped

  def get_aggregated_groups(self, extra_aggregated_groups=[] ):
    return remove_repeats( self.aggregated_groups + extra_aggregated_groups )

  def get_aggregated_keys(self, extra_aggregating_keys=[] ):
    return remove_repeats( self.aggregating_keys + extra_aggregating_keys )
