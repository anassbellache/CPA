import torch
import torch.optim as optim
from torch import nn
import json
from model import NBLoss


class BaselineMLPModel(nn.Module):
    def __init__(self, num_genes, num_drugs, num_covariates, embedding_dim, hidden_units, device="cuda", hparams="",
                 loss="gauss"):
        super(BaselineMLPModel, self).__init__()
        self.hparams = hparams
        self.num_genes = num_genes
        self.device = device
        self.loss_baseline = loss

        self.hparams = self.set_hparams_(hparams)

        # Define the embedding layers
        self.gene_embedding = nn.Embedding(num_genes, self.hparams["dim"])
        self.drug_embedding = nn.Embedding(num_drugs, self.hparams["dim"])
        self.covariate_embedding = nn.Embedding(num_covariates, self.hparams["dim"])

        # Define the MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.hparams["dim"] * 3, self.hparams["mlp_width"]),  # times 3 because we have 3 embeddings
            nn.ReLU(),
            *[
                nn.Sequential(
                    nn.Linear(self.hparams["mlp_width"], self.hparams["mlp_width"]),
                    nn.ReLU(),
                )
                for _ in range(self.hparams["mlp_depth"] - 1)
            ],
            nn.Linear(self.hparams["mlp_width"], 2 * num_genes),
        )

        # Define the optimizer
        self.optimizer = optim.Adam(
            self.parameters(),
            lr=self.hparams["mlp_lr"],
            weight_decay=self.hparams["mlp_wd"],
        )

        if self.loss_baseline == "nb":
            self.loss_baseline = NBLoss()
        elif self.loss_baseline == 'gauss':
            self.loss_baseline = nn.GaussianNLLLoss()

    def forward(self, genes, drugs, covariates):
        gene_embedding = self.gene_embedding(genes)
        drug_embedding = self.drug_embedding(drugs)
        covariate_embedding = self.covariate_embedding(covariates)

        # Concatenate the embeddings and pass through MLP
        x = torch.cat([gene_embedding, drug_embedding, covariate_embedding], dim=-1)
        gene_reconstructions = self.mlp(x)

        # Assuming the first half of the output are the means and the second half are the variances
        gene_means = gene_reconstructions[:, :self.num_genes]
        gene_vars = gene_reconstructions[:, self.num_genes:]

        return gene_means, gene_vars

    def set_hparams_(self, hparams):
        """
        Set hyper-parameters to default values or values fixed by user for those
        hyper-parameters specified in the JSON string `hparams`.
        """

        self.hparams = {
            "dim": 128,
            "mlp_width": 128,
            "mlp_depth": 2,
            "mlp_lr": 4e-3,
            "mlp_wd": 1e-7,
            "batch_size": 256,
            "step_size_lr": 45,
        }

        # the user may fix some hparams
        if hparams != "":
            if isinstance(hparams, str):
                self.hparams.update(json.loads(hparams))
            else:
                self.hparams.update(hparams)

        return self.hparams

    def move_inputs_(self, genes, drugs, covariates):
        """
        Move minibatch tensors to CPU/GPU.
        """
        if genes.device.type != self.device:
            genes = genes.to(self.device)
            if drugs is not None:
                drugs = drugs.to(self.device)
            if covariates is not None:
                covariates = [cov.to(self.device) for cov in covariates]
        return genes, drugs, covariates

    def update(self, genes, drugs, covariates):
        """
        Update MLP's parameters given a minibatch of genes, drugs, and covariates.
        """
        genes, drugs, covariates = self.move_inputs_(genes, drugs, covariates)

        # Forward pass through the model
        gene_means, gene_vars = self.predict(genes, drugs, covariates)

        # Compute the reconstruction loss
        reconstruction_loss = self.loss_baseline(gene_means, genes, gene_vars)

        # Zero the gradients
        self.optimizer.zero_grad()

        # Backward pass (compute gradients)
        reconstruction_loss.backward()

        # Update the weights
        self.optimizer.step()

        # Increment the iteration counter
        self.iteration += 1

        return {
            "loss_reconstruction": reconstruction_loss.item(),
        }