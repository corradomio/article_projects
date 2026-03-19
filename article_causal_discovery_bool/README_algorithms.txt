GAE
    train: epochs
        train_step: update_freq


PNL:
    scandisce tutte le coppie di variabili
        per ogni coppia
            l1 = mlp(input_dim=1, hidden_layers=self.hidden_layers, hidden_units=self.hidden_units, output_dim=1)
            l2 = mlp(input_dim=1, hidden_layers=self.hidden_layers, hidden_units=self.hidden_units, output_dim=1)
            e = nonlinear_ica(l1, l2)

Lingam
    ICA-based LiNGAM
    DirectLiNGAM
    Pairwise LiNGAM
    Hidden common causes