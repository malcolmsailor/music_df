import pandas as pd

# Ensure consistent DataFrame display across different terminal widths (e.g., when
# running from Claude Code vs interactive terminal)
pd.set_option("display.width", 1000)
pd.set_option("display.max_columns", None)
pd.set_option("display.expand_frame_repr", False)
