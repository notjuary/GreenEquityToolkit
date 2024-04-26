
class ModelManager:

    def __init__(self):
        self.models = {}


    def addModel(self, name,precision_base,recall_base,f1_score_base,mean_base,median_base, variance_base, overall_accuracy_base, energy_consumption_graph,combined_energy_graph,metrics_graph,accuracy_graph):
        self.models[name] = {
            'precision_base':precision_base,
            'recall_base':recall_base,
            'f1_score_base':f1_score_base,
            'mean_base':mean_base,
            'median_base':median_base,
            'variance_base':variance_base,
            'overall_accuracy_base':overall_accuracy_base,
            'energy_consumption_graph':energy_consumption_graph,
            'combined_energy_graph':combined_energy_graph,
            'metrics_graph':metrics_graph,
            'accuracy_graph':accuracy_graph
        }

    def getPrecision(self,name):
        return self.models[name]['precision_base']

    def getRecall(self,name):
        return self.models[name]['recall_base']

    def getF1(self,name):
        return self.models[name]['f1_score_base']

    def getMean(self,name):
        return self.models[name]['mean_base']

    def getMedian(self,name):
        return self.models[name]['median_base']

    def getVariance(self,name):
        return self.models[name]['variance_base']

    def getOverallAccuracy(self,name):
        return self.models[name]['overall_accuracy_base']

    def getEnergyConsumptionGraph(self,name):
        return self.models[name]['energy_consumption_graph']

    def getCombinedEnergyConsumptionGraph(self,name):
        return self.models[name]['combined_energy_graph']

    def getMetricsGraph(self,name):
        return self.models[name]['metrics_graph']

    def getAccuracyGraph(self,name):
        return self.models[name]['accuracy_graph']

    def __str__(self):
        return str(self.models)