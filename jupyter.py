# dumb setup stuff
from IPython.core.display import display, HTML
import plotly.io as pio
import plotly.graph_objects as go
import pandas

def setup():
  display(HTML("<style>.container { width:100% !important; }</style>"))
  # matplotlib.rcParams['figure.figsize'] = [8, 5]

  # pandas.set_option("display.width", 10000)
  # pandas.set_option("display.max_colwidth", None)
  pandas.set_option("display.max_rows", 100)

  pio.templates["ians_template"] = go.layout.Template(
    layout=go.Layout(
      height=800,
      font=dict(
        size=16
      )
    )
  )

  pio.templates.default = "ians_template"
