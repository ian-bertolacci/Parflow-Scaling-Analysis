import functools, itertools, re, pandas, os, operator, time

def wrapped_range(first, start, end):
  if not ( start <= first and first <= end ):
    raise ValueError( f"Wrapped range, the following must be true: start <= first <= end ({start} {'<=' if start <= first else '<!='} {first} {'<=' if first <= end else '<!='} {end})" )
  return itertools.chain( range(first, end), range(start,first) )

def shifted_product( *iterables ):
  list_iterables = [ list(iterable) for iterable in iterables ]
  for shift in itertools.product( *(range(len(l)) for l in list_iterables) ):
    for tup in zip( *( [ list_iterables[d][i] for i in wrapped_range(shift[d], 0, len(list_iterables[d])) ] for d in range(len(list_iterables)) ) ):
      yield tup

# is safe for use with generators
def isiterable(obj):
  try:
    iter(obj)
  except Exception:
    return False
  else:
    return True

def my_pprint( stuff ):
  print( json.dumps(stuff, indent=1) )

def indent_string( string, indentation="\t", depth=1  ):
  indent = (indentation*depth)
  return "\n".join( [ f"{indent}{line}" for line in string.split("\n") ] )

def join_lines_strings( *strings, indentation="\t", depth=0 ):
  indent = (indentation*depth)
  return "\n".join( [ f"{indent_string(string, indentation=indentation, depth=depth)}" for string in strings ] ) #indent + ("\n"+indent).join( strings )

def remove_repeats( list_with_possible_duplicates ):
  seen = []
  list_without_duplicates = []
  for i in list_with_possible_duplicates:
    if i not in seen:
      list_without_duplicates.append( i )
      seen.append( i )
  return list_without_duplicates


def align_integers( *args, spacing=" " ):
  strings = [ f"{arg}" for arg in args ]
  lengths = [ len(string) for string in strings ]
  max_length = max( *lengths )
  strings = [ f"{spacing*(max_length-length)}{string}" for (string, length) in zip(strings, lengths) ]
  return strings

def align_decimals(*args,  precision=None, spacing=" ", comma=True, align_leading=True, align_trailing=True ):
  comma = "," if comma else ""

  if precision != None:
    if type(precision) == int:
      precision = [ precision ] * len(args)
    strings = [ f"{arg:{comma}.{prec}f}" for (arg,prec) in zip(args,precision) ]
  else:
    strings = [ f"{arg:{comma}}" for arg in args ]

  if     (not ( align_leading or align_trailing ) ) \
      or len(args) <= 1:
    return strings

  splits = [ string.split(".") for string in strings ]

  leadings = [ len(split[0]) for split in splits ]
  trailings = [ len(split[1]) for split in splits ]

  max_leading = max( *leadings )
  max_trailing = max( *trailings )

  new_strings = [ string for string in strings ]

  if align_leading:
    new_strings = [
      f"{spacing*(max_leading-leading)}{string}"
      for (string, leading) in zip(new_strings, leadings)
    ]
  if align_trailing:
   new_strings = [
     f"{string}{spacing*(max_trailing-trailing)}"
     for (string, trailing) in zip(new_strings, trailings)
   ]
  if align_trailing and not align_leading:
    new_strings = [
     f"{string}{spacing*(max_leading-leading)}"
     for (string, leading) in zip(new_strings, leadings)
   ]

  return new_strings

def iterdirs( directory, matches=re.compile(".*", flags=re.DOTALL) ):
  return ( name for name in os.listdir(directory) if os.path.isdir( os.path.join( directory, name ) ) and matches.match(name) != None )

def iterfiles( directory, matches=re.compile(".", flags=re.DOTALL ) ):
  return ( name for name in os.listdir(directory) if os.path.isfile( os.path.join( directory, name ) ) and matches.match(name) != None )

def proper_type_dict( dictionary ):
  def bool_str_cast( str_value ):
    s = str_value.lower()
    if s not in ["true","false"]:
      raise ValueError( f"{s} is not a boolean string" )
    return s == "true"

  def cast( str_value ):
    if str_value == None:
      return "None"
    for cast_fn in [int, float, bool_str_cast, str]:
      try:
        cast_value = cast_fn( str_value )
        return cast_value
      except ValueError:
        pass

  return { key: cast(value) for key,value in dictionary.items() }

def get_set( dataframe, *columns ):
  return dataframe[ dataframe.duplicated( subset=columns, keep="first" ) == False ][list(columns)]

def get_actual_set( dataframe, *columns ):
  if isinstance(dataframe, pandas.DataFrame):
    return set( ( ( tuple( k ) if len(k) > 1 else tuple(k)[0]) for k in get_set( dataframe, *columns ).values ) )
  else:
    return set( dataframe )

def recursive_unpack( obj, do_not_unpack_when_true=lambda x: False ):
  bad_types = [str]
  if functools.reduce( operator.or_, map( lambda t: isinstance(obj, t), bad_types ), False ):
    yield obj
  elif do_not_unpack_when_true( obj ):
    yield obj
  else:
    # If object is an iterable, need to unpack its elements
    if isiterable(obj):
      # Some iterables we want to be careful about
      if isinstance(obj, dict):
        iterable = obj.values()
      # others it doesnt matter
      else:
        iterable = obj
      # Iterate over all the iterable's elements
      for possiblely_nested_obj in iterable:
        # Unpack the possibly nested values inside that element
        for fully_unpacked_obj in recursive_unpack( possiblely_nested_obj, do_not_unpack_when_true ):
          yield fully_unpacked_obj
    # If object is not iterable, we have successfully reached an inner-most nested value and simply yield it
    else:
      yield obj
