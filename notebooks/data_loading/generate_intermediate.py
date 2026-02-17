# %% [markdown]
# # Intermediate Frame Generation
# ## Modified: March 25th, 2024
#
# This notebook crawls the Brain Radio data directory and populates a (custom) dataframe with the frequency features specified.
#
brainradio_data_directory = "/home/virati/Data/phd_vrt_2013/neural/lfp"
clinical_data_directory = "/home/virati/Data/phd_vrt_2013/clinical"
do_saves = True

# %%
# %reload_ext autoreload
# %autoreload 2
from pathlib import Path
from datetime import date
import logging
from dbspace.readout.BR_DataFrame import BR_Data_Tree

brainradio_data_directory = Path(brainradio_data_directory)
clinical_data_directory = Path(clinical_data_directory)

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

# %%
# # Input Specifications
# You need to specify the parent directory for the data.
#
# You should also specify the feature sets you want to include in your analysis.

# %%

# %%
DataFrame = BR_Data_Tree(
    clin_vector_file=clinical_data_directory / "clinical_vectors.json",
    input_data_directory=brainradio_data_directory,
)
DataFrame.run_loading().Save_Frame(name_addendum="Feb2025_F")


# %%
# This loads in the TimeDomain
# DataFrame.generate_TD_sequence()
# if do_saves:
#    DataFrame.Save_Frame(name_addendum="Feb2026_TD")

# %%
# last successful run
today = date.today()
print(f"Last Successful Run: {today}")
