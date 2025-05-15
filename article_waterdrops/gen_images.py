from joblib import Parallel, delayed
from waterdrop import render_water_drop


SCENE_NAME = "drop_scene_v2"


def main():
    # prepare the parameters
    args_list: list[dict] = []
    for contat_angle in list(range(1,91, 3)):
        args_list.append(dict(
            scene_name=SCENE_NAME,
            camera_shift_z=0,
            scene_shift_z=1,

            drop_base=2,
            drop_height=1,
            drop_shift=0,

            table_rot_z=0,
            table_shift_z=0,

            dispenser_side=1,
            dispenser_shift_z=0,
        ))

    for args in args_list:
        render_water_drop(**args)
        break

    # Parallel(n_jobs=14)(delayed(render_water_drop)(**args) for args in args_list)
# end


if __name__ == "__main__":
    main()

