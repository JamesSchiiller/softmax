virtualenv -p python3.6 venv
source venv/bin/activate

pip install numpy
pip install keras
pip install tensorflow
pip install sklearn
pip install matplotlib
pip install scipy <~~~ you don't need this but doesn't hurt

python x.py

example inputs and output prediction 
====================================
INPUT:
prediction = model.predict_proba(array([[-11.89337759, -13.65864154]]))
OUTPUT:
[[9.6482694e-01 4.8628806e-11 3.5173021e-02]] 
That is, there is a 96.482% probability of it being class 1 
and much lower probability of it being one of the other two classes 
(96.482%, 0.00000000001%, 3.517%)

References
https://machinelearningmastery.com/snapshot-ensemble-deep-learning-neural-network/