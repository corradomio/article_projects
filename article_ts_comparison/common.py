import h5py
import stdlib.csvx as csvx
from datetime import datetime
from sklearnx.metrics import weighted_absolute_percentage_error
from stdlib.tprint import tprint

RESULTS_WAPE = [["lib", "item_area", "model", "wape"]]
PREDICT_FILE = "predict_models.hdf5"


# ---------------------------------------------------------------------------

def use_model(g, name, model, Xtr_scaled, ytr_scaled, Xte_scaled, yte_true, fh, USED_LIBRARY, RESULT_FILE):
    start = datetime.now()
    item_area: str = g[0]
    try:
        ianame = item_area.replace('/', '_').replace(' ', '_')
        tprint(f"... {item_area}/{name}")

        # train
        model.fit(y=ytr_scaled, X=Xtr_scaled)
        # predict
        yte_predicted = model.predict(fh, X=Xte_scaled)
        # done

        time = (datetime.now()-start).seconds
        wape = weighted_absolute_percentage_error(yte_true, yte_predicted)

        RESULTS_WAPE.append([USED_LIBRARY, item_area, name, wape])
        csvx.save_csv(RESULT_FILE, RESULTS_WAPE[1:], header=RESULTS_WAPE[0])

        PREDICT_FILE = f"{USED_LIBRARY}-predictions.hdf5"
        with h5py.File(PREDICT_FILE, mode="a") as df:

            if f"{ianame}/true" not in df:
                ytrue = yte_true.values
                dstrue = df.create_dataset(f"{ianame}/true", shape=ytrue.shape, data=ytrue)

            ypred = yte_predicted.values
            dspred = df.create_dataset(f"{ianame}/{USED_LIBRARY}/{name}", shape=ypred.shape, data=ypred)
            dspred.attrs['wape'] = wape
            dspred.attrs['time'] = time
        # end
    except Exception as e:
        tprint(f"ERROR: {item_area}/{name}: {e}")
        pass
    return
# end

