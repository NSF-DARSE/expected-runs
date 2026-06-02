#import sys
import glob

#path = "../python_files/"
#sys.path.append(path)

from expected_runs.generate_gamestate_summary import *
 

base_path = "../../v3"
years = ["2024", "2025"]
out_dir = "../python_files/results/CSV_files"

summary_df = build_gamestate_summary_all_years(base_path, out_dir)

summary_path = glob.glob(out_dir + "GameState_*.csv")

print(summar_path)

"""
final_df = build_final_dataset(
                base_path,
                years=["2024", "2025"],
                summary_path,
                out_dir,
                save=True)
"""
