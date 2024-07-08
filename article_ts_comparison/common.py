import h5py
import stdlib.csvx as csvx
from sklearnx.metrics import weighted_absolute_percentage_error
from stdlib.tprint import tprint

RESULTS_WAPE = [["lib", "item_area", "model", "wape"]]
# USED_LIBRARY = "unk"
# RESULT_FILE = "wape_skt_nn.csv"
PREDICT_FILE = "predict_models.hdf5"


# ---------------------------------------------------------------------------

def use_model(g, name, model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_true, fh, USED_LIBRARY, RESULT_FILE):
    item_area = g[0]
    try:
        tprint(f"... {item_area}/{name}")

        model.fit(y=ytr_scaled, X=Xtr_scaled)
        yte_predicted = model.predict(fh, X=Xte_scaled)

        wape = weighted_absolute_percentage_error(yte_true, yte_predicted)

        RESULTS_WAPE.append([USED_LIBRARY, item_area, name, wape])
        csvx.save_csv(RESULT_FILE, RESULTS_WAPE[1:], header=RESULTS_WAPE[0])

        with h5py.File(PREDICT_FILE, mode="w") as df:
            ydata = ytr_scaled.values
            name = f"/{USED_LIBRARY}/{model}"
            dset = df.create_dataset(name, shape=ydata.shape, data=ydata)
            dset.attrs['wape'] = wape
        # end
    except Exception as e:
        tprint(f"ERROR: {item_area}/{name}: {e}")
        pass
    return
# end

