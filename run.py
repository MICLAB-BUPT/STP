from stp.trainer import Trainer


# ------------- initialize distribution ------------- #
"""
    ...
"""

# ------------- initialize model ------------- #

model = ...

# ------------- initialize trainer ------------- #
trainer = Trainer(model, ...)

# ------------- training loop ------------- #
num_epochs = 10
for epoch in range(num_epochs):
    trainer.step()