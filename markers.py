import plotly, functools
from plotly.validators.scatter.marker import SymbolValidator

from analysis.simple_factories import *

css_colors= [
"aliceblue", "antiquewhite", "aqua", "aquamarine", "azure", "beige", "bisque", "black", "blanchedalmond", "blue",
"blueviolet", "brown", "burlywood", "cadetblue", "chartreuse", "chocolate", "coral", "cornflowerblue",
"cornsilk", "crimson", "cyan", "darkblue", "darkcyan", "darkgoldenrod", "darkgray", "darkgrey", "darkgreen",
"darkkhaki", "darkmagenta", "darkolivegreen", "darkorange", "darkorchid", "darkred", "darksalmon", "darkseagreen",
"darkslateblue", "darkslategray", "darkslategrey", "darkturquoise", "darkviolet", "deeppink", "deepskyblue",
"dimgray", "dimgrey", "dodgerblue", "firebrick", "floralwhite", "forestgreen", "fuchsia", "gainsboro",
"ghostwhite", "gold", "goldenrod", "gray", "grey", "green", "greenyellow", "honeydew", "hotpink", "indianred", "indigo",
"ivory", "khaki", "lavender", "lavenderblush", "lawngreen", "lemonchiffon", "lightblue", "lightcoral", "lightcyan",
"lightgoldenrodyellow", "lightgray", "lightgrey", "lightgreen", "lightpink", "lightsalmon", "lightseagreen",
"lightskyblue", "lightslategray", "lightslategrey", "lightsteelblue", "lightyellow", "lime", "limegreen",
"linen", "magenta", "maroon", "mediumaquamarine", "mediumblue", "mediumorchid", "mediumpurple",
"mediumseagreen", "mediumslateblue", "mediumspringgreen", "mediumturquoise", "mediumvioletred", "midnightblue",
"mintcream", "mistyrose", "moccasin", "navajowhite", "navy", "oldlace", "olive", "olivedrab", "orange", "orangered",
"orchid", "palegoldenrod", "palegreen", "paleturquoise", "palevioletred", "papayawhip", "peachpuff", "peru", "pink",
"plum", "powderblue", "purple", "red", "rosybrown", "royalblue", "rebeccapurple", "saddlebrown", "salmon",
"sandybrown", "seagreen", "seashell", "sienna", "silver", "skyblue", "slateblue", "slategray", "slategrey", "snow",
"springgreen", "steelblue", "tan", "teal", "thistle", "tomato", "turquoise", "violet", "wheat", "white", "whitesmoke",
"yellow", "yellowgreen"
]

all_css_usable_colors = sorted( [
  color
  for color in css_colors
  if functools.reduce(
      lambda a,b,: a and b,
      map(
        lambda x: x not in color,
        ["white", "gray", "pale", "tan", "azure", "beige", "bisque", "alice", "cornsilk", "blanch", "blush", "cream", "wheat", "snow", "light", "gainsboro", "ivory", "khaki", "lavender", "honeydew", "linen", "lemonchiffon", "mistyrose", "moccasin", "oldlace", "whip", "seashell"]
      )
    ) \
    and "light" not in color and "dark" not in color
] )

all_usable_colors = plotly.colors.qualitative.G10

def get_usable_colors( ):
  return [
    color for color in all_usable_colors
  ]

def get_usable_colors_without( *without ):
  return [
    color for color in all_usable_colors
    if color not in without
  ]

def get_usable_colors_without_rx( *without ):
  color_re = re.compile( r"|".join( (f"({color})" for color in without) ) )
  return [
    color for color in all_usable_colors
    if color_re.search( color ) == None
  ]

def demo_colors( colors=get_usable_colors() ):
  plots = []

  for (idx, color) in enumerate( colors ):
    plots.append(
      go.Scattergl(
        x=[idx],
        y=[idx],
        name=color,
        mode="markers",
        marker=dict( color=color, size=10 )
      )
    )

  fig = go.Figure(
    data=plots,
    layout=dict(
      autosize=True,
      height=1024,
    )
  )

  fig.show()


all_base_marker_symbols = [
  symbol_string
  for symbol_string in plotly.validators.scatter.marker.SymbolValidator().values[1::2]
  if not any( variant in symbol_string for variant in ["-open", "-dot"] )
]

all_base_marker_symbols_ordered = [
  "circle",

  "square",
  "cross",
  "star-triangle-up",

  "diamond",
  "x",
  "star-triangle-down",

  "star-square",
  "star",

  "star-diamond",
#   "asterisk", # Does not have a fill, so appears invisible
]

all_usable_marker_symbols = all_base_marker_symbols_ordered

def get_usable_marker_symbols( reverse=True ):
  list_copy =  [ symbol_string for symbol_string in all_usable_marker_symbols ]
  if reverse:
    list_copy.reverse()
  return list_copy

def demo_symbols( symbols=get_usable_marker_symbols() ):
  namestems = []
  namevariants = []
  symbols = []
  for symbol in symbols:
      name = symbol
      symbols.append(symbol)
      namestems.append(name.replace("-open", "").replace("-dot", ""))
      namevariants.append( name.replace(name,"") )
  print(symbols)
  fig = go.Figure(go.Scattergl(mode="markers", x=namevariants, y=namestems, marker_symbol=symbols,
                             marker_line_color="midnightblue", marker_color="lightskyblue",
                             marker_line_width=2, marker_size=15,
                             hovertemplate="name: %{y}%{x}<br>number: %{marker.symbol}<extra></extra>"))
  fig.update_layout(title="Mouse over symbols for name & number!",
                    xaxis_range=[-1,4], yaxis_range=[len(set(namestems)),-1],
                    margin=dict(b=0,r=0), xaxis_side="top", height=1200, width=400)
  fig.show()


class UniqueColorFactory(UniqueFactory):
  def __init__(self, initial=all_usable_colors, *args, **kwargs):
    super().__init__( initial=initial, *args, **kwargs )
class RotatingColorFactory(RotatingFactory):
  def __init__(self, initial=all_usable_colors, *args, **kwargs):
    super().__init__( initial=initial, *args, **kwargs )


class UniqueSymbolFactory(UniqueFactory):
  def __init__(self, initial=all_usable_marker_symbols, *args, **kwargs):
    super().__init__( initial=initial, *args, **kwargs )
class RotatingSymbolFactory(RotatingFactory):
  def __init__(self, initial=all_usable_marker_symbols, *args, **kwargs):
    super().__init__( initial=initial, *args, **kwargs )


class UniqueColorSymbolFactory(UniqueProductFactory):
  def __init__(self, collections=[ all_usable_colors, all_usable_marker_symbols ], *args, **kwargs):
    super().__init__( collections=collections, *args, **kwargs )
class RotatingColorSymbolFactory(RotatingProductFactory):
  def __init__(self, collections=[ all_usable_colors, all_usable_marker_symbols ], *args, **kwargs):
    super().__init__( collections=collections, *args, **kwargs )
