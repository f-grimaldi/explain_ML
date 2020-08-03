import numpy as np
import shap

"""
Class based on methods found on: https://github.com/slundberg/shap
"""
class SHAP():

    def __init__(self, model, model_type, data):
        """
        Args:
        Name            Type                    Descr
        model           sklearn/keras/pytorch   Target model
        model_type      str                     The type of model we are using (Tree, Gradient, Kernel)
        data            pandas.DataFrame        The input matrix
        """
        self.model = model
        self.data = data
        self.type = model_type


    def get_explainer(self, **args):
        raise NotImplementedError


    """
    Visualize the results of SHapley Additive exPlanations (SHAP) tecnique on a given example
    https://www.kaggle.com/learn/machine-learning-explainability
    """
    def plot_shap(self, nrow, **args):
        """
        Args:
        nrow            int/list             The number of the example subject to the SHAP analysis
        """
        # 1. Define input
        input = data[nrow]
        # 2. Create object that can calculate shap values
        explainer = self.get_explainer(**args)
        # 3. Calculate Shap values
        shap_values = explainer.shap_values(input)
        # 4. Return plot
        shap.initjs()
        return shap.force_plot(explainer.expected_value[1], shap_values[1], input)





if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import load_iris
    from sklearn.metrics import accuracy_score

    data = load_iris()
    X, y = data['data'], data['target']

    model = RandomForestClassifier()
    model.fit(X, y)
    plot_shap(model, X, 2)
    plt.show()
