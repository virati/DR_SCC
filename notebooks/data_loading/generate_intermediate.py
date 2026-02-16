# %% [markdown]
# # Intermediate Frame Generation
# ## Modified: March 25th, 2024
#
# This notebook crawls the Brain Radio data directory and populates a (custom) dataframe with the frequency features specified.
#

# %%
# %reload_ext autoreload
# %autoreload 2
from pathlib import Path

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)


# %%

import dbspace
from dbspace.readout.BR_DataFrame import BR_Data_Tree

# %% [markdown]
# # Input Specifications
# You need to specify the parent directory for the data.
#
# You should also specify the feature sets you want to include in your analysis.

# %%

brainradio_data_directory = Path(
    "/home/virati/Data/phd_vrt_2013/neural/lfp"
)  # suggest this directory inside the devcontainer
clinical_data_directory = Path("/home/virati/Data/phd_vrt_2013/clinical")


# %%
DataFrame = BR_Data_Tree(
    clin_vector_file=clinical_data_directory / "clinical_vectors.json",
    input_data_directory=brainradio_data_directory,
)
DataFrame.run_loading().Save_Frame()


# %%

DataFrame.generate_TD_sequence()

# %%
DataFrame.Save_Frame(name_addendum="March2024")

# %%
# last successful run
from datetime import date

today = date.today()
print(f"Last Successful Run: {today}")
