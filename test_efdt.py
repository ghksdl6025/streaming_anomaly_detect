from numpy.random import seed
from river import synth
from river import evaluate
from river import metrics
from river import tree

gen = synth.Agrawal(classification_function=0, seed=42)
dataset =  iter(gen.take(1000))

model = tree.ExtremelyFastDecisionTreeClassifier(
    grace_period=200,
    split_confidence=1e-5,
    # nominal_attributes=[]
    min_samples_reevaluate=100
)
metric =  metrics.Accuracy()

for x,y in dataset:
# print(evaluate.progressive_val_score(dataset,model, metric))
    y_predict = model.predict_one(x)
    model.learn_one(x,y)
    print(model.predict_proba_one(x))
    metric.update(y, y_predict)
    print(metric)