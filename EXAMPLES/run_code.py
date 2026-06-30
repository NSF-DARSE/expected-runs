#import sys
import glob
import os
from sklearn.ensemble import RandomForestRegressor

#path = "../python_files/"
#sys.path.append(path)

from expected_runs.generate_gamestate_summary import *

from expected_runs.filter_pitch_type import *
 
# /path/to/trackman/data
data_path = "../../v3"
years = ["2024", "2025"]

OUT_DIR = "../results/CSV_files"

summary_df = build_gamestate_summary_all_years(data_path, OUT_DIR)


# returns a list of gamestate.csv files choose one in next fun
summary_files = sorted(glob.glob(OUT_DIR + "GameState_*.csv"),
                        key=os.path.getmtime, reverse=True)                                                                            

target_df = build_final_dataset(
                data_path,
                years=["2024", "2025"],
                summary_files[0],
                OUT_DIR,
                save=True)

# getting file paths
ff_output_path, team_output_path = get_timestamped_output_paths(OUT_DIR)

target_files = sorted(glob.glob(OUT_DIR + "/Final_Target_Calc_*.csv")
                        key=os.path.getmtime, reverse=True)                                                                            

final_ff_df = create_ff_dataset(input_path=target_files[0], 
                            output_path=ff_output_path)

team_ff_df = create_team_ff_dataset(df_ff = final_ff_df,
                                    output_path=team_output_path)

input_paths = sorted(glob.glob(out_dir + "/df_ff_*.csv")
                        key=os.path.getmtime, reverse=True)

rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
)

model, train_results, test_results = train_model(
                        input_path = input_paths[0], 
                        model_path = out_dir+f"/model_{timestamp}.csv",
                        model   = rf, test_size=0.2,
                        random_state=42)                          
