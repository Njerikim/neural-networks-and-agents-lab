# Neural Networks & Agents Lab

Three short, runnable demos in one notebook: a neural network for credit risk, a model-explainability check, and a small autonomous agent for water-pump maintenance. All three run on synthetic data, so the notebook is self-contained and needs no downloads.

## What's inside

### 1. Credit risk neural network (PyTorch)
A feed-forward network that outputs the probability of a risky applicant.

- Architecture: 4 inputs → 16 → 8 → 1, with ReLU, 20% dropout and a sigmoid output
- Training: 500 synthetic samples, 80/20 split, binary cross-entropy loss, Adam optimiser (learning rate 0.01), 50 epochs
- Result: training loss falls to **0.0605**

### 2. Model explainability (scikit-learn)
A random forest ranks which inputs drive a credit decision, using features relevant to Kenyan lending:

| Feature | Importance |
|---|---|
| Monthly income | 0.5310 |
| Saving ratio | 0.2644 |
| Mobile money volume | 0.1160 |
| Informal credit score | 0.0886 |

### 3. Autonomous maintenance agent
A perceive → reason → act loop for a rural water pump (`PUMP-KE-042`, Embu).

1. **Perceive:** read acoustic telemetry (vibration anomaly score 0.87)
2. **Reason:** compare the score to a 0.75 threshold and decide the pump is at risk of failing within 48 hours
3. **Act:** log a maintenance ticket with location and reason, and notify the field team

## Run it

```bash
pip install numpy pandas torch scikit-learn jupyter
jupyter notebook deeplearning.ipynb
```

## Limitations

- **Synthetic data.** The labels come from simple rules, so the feature ranking in demo 2 reflects how the data was generated, not real credit behaviour.
- **Training loss only.** Demo 1 reports loss on the training set. The test split is created but not yet used for evaluation.
- **Simulated agent.** The telemetry is hard-coded and the "SMS notification" is printed, not sent. It shows the agent pattern, not a deployed system.

## Next steps

- Evaluate the network on the held-out set (accuracy, AUC)
- Replace the synthetic data with a real dataset
- Connect the agent to live sensor data and a real SMS provider

## Author

Beth Njeri Kimani, AI/ML engineer and founder of [Axene.io](https://axene.io). Nairobi, Kenya.
