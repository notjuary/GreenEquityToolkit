import csv
import os
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go

output_dir = '.'
output_file = 'emissions.csv'


class ModelManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ModelManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance
    def __init__(self):
        self.figs=[]
        self.models = {}
        self.run_id_to_model = {}

    def get_last_run_id(self, csv_file_path):
        df = pd.read_csv(csv_file_path)
        last_run_id = df.iloc[-1]['run_id']  # Get the run_id of the last row
        return last_run_id

    def addModel(self, name, accuracy, precision_base, recall_base, f1_score_base, mean_base, mean_difference_base,
                 equal_opportunity_difference, average_odds_difference):
        csv_file_path = os.path.join(output_dir, output_file)
        if os.path.isfile(csv_file_path):
            print(f"CSV file created: {csv_file_path}")

        energy_metrics = self.readEnergyMetricsFromCSV()
        self.models[name] = {
            'accuracy': accuracy,
            'precision_base': precision_base,
            'recall_base': recall_base,
            'f1_score_base': f1_score_base,
            'mean_base': mean_base,
            'mean_difference_base': mean_difference_base,
            'equal_opportunity_difference': equal_opportunity_difference,
            'average_odds_difference': average_odds_difference,
            'energy_metrics': energy_metrics
        }
        run_id = self.get_last_run_id(csv_file_path)
        self.run_id_to_model[run_id] = name
        self.updateGraphs()

    def getPrecision(self, name):
        return self.models[name]['precision_base']

    def getRecall(self, name):
        return self.models[name]['recall_base']

    def getF1(self, name):
        return self.models[name]['f1_score_base']

    def getMean(self, name):
        return self.models[name]['mean_base']

    def getMeanDifference(self, name):
        return self.models[name]['mean_difference_base']

    def getEqualOpportunityDifference(self, name):
        return self.models[name]['equal_opportunity_difference']

    def getAverageOddsDifference(self, name):
        return self.models[name]['average_odds_difference']

    def updateGraphs(self):
        dataResults = []
        for name, model_data in self.models.items():
            energy_metrics = model_data['energy_metrics']
            dataResults.append({
                'model_name': name,
                'accuracy': model_data['accuracy'],
                'precision_base': model_data['precision_base'],
                'recall_base': model_data['recall_base'],
                'f1_score_base': model_data['f1_score_base'],
                'mean_base': model_data['mean_base'],
                'mean_difference_base': model_data['mean_difference_base'],
                'equal_opportunity_difference': model_data['equal_opportunity_difference'],
                'average_odds_difference': model_data['average_odds_difference'],
                'energy_consumed': energy_metrics['energy_consumed'],
                'duration': energy_metrics['duration'],
                'ram_energy': energy_metrics['ram_energy'],
                'cpu_energy': energy_metrics['cpu_energy'],
                'gpu_energy': energy_metrics['gpu_energy']
            })

        df = pd.DataFrame(dataResults)

        figs = []
        for metric_name, y_column in [("Energy Consumption Distribution", "energy_consumed"),
                                      ("Duration Distribution", "duration"),
                                      ("RAM Energy Distribution", "ram_energy"),
                                      ("CPU Energy Distribution", "cpu_energy"),
                                      ("GPU Energy Distribution", "gpu_energy")]:
            fig = go.Figure()


            for model_name, model_data in self.models.items():
                fig.add_trace(go.Scatter(
                    x=[model_name],
                    y=[model_data['energy_metrics'][y_column]],
                    mode='markers',
                    name=model_name,
                    text=[model_data['energy_metrics'][y_column]],
                    hovertemplate='%{text}',
                ))

                fig.update_layout(
                    title=metric_name,
                    xaxis_title='Model Name',
                    yaxis_title=f"{metric_name} ({'Joules' if 'energy' in y_column else 'seconds'})",
                )

            figs.append(fig.to_json())

        for metric_name in ["accuracy", "precision_base", "recall_base", "f1_score_base", "mean_base",
                            "mean_difference_base", "equal_opportunity_difference", "average_odds_difference"]:
            formatted_metric_name = metric_name.replace('_', ' ').title()
            fig = px.histogram(df, x="model_name", y=metric_name, title=formatted_metric_name+" Distribution",labels={"model_name": "Model Name", metric_name: formatted_metric_name})  # Personalizza le etichette degli assi
            fig.update_xaxes(title="Model Name")
            fig.update_traces(hovertemplate=f'{formatted_metric_name}: %{{y:.2f}}')
            fig.update_yaxes(title=formatted_metric_name)
            figs.append(fig.to_json())

        self.figs = figs

    def readEnergyMetricsFromCSV(self):
        last_run_metrics = None
        csv_file_path = os.path.join(output_dir, output_file)
        if os.path.isfile(csv_file_path):
            print(f"CSV file created: {csv_file_path}")

            with open(csv_file_path, 'r') as csvfile:
                csv_reader = csv.DictReader(csvfile, delimiter=',')
                for row in csv_reader:
                    # Convert kWh to Joules
                    energy_in_joules = float(row['energy_consumed']) * 3600000  # Conversion factor (1 kWh = 3600000 J)
                    ram_energy_in_joules = float(row['ram_energy']) * 3600000
                    cpu_energy_in_joules = float(row['cpu_energy']) * 3600000
                    gpu_energy_in_joules = float(row['gpu_energy']) * 3600000
                    duration=float(row['duration']) # Convert duration to float
                    model_name = self.run_id_to_model.get(row['run_id'],
                                                          'Unknown')  # Get the model name from the run_id

                    # Check if this is the last run_id
                    if row['run_id'] == self.get_last_run_id(csv_file_path):
                        last_run_metrics = {
                            'timestamp': row['timestamp'],
                            'run_id': row['run_id'],
                            'model_name': model_name,
                            'energy_consumed': energy_in_joules,
                            'duration': duration,
                            'ram_energy': ram_energy_in_joules,
                            'cpu_energy': cpu_energy_in_joules,
                            'gpu_energy': gpu_energy_in_joules
                        }

        else:
            print(f"CSV file not found: {csv_file_path}")

        return last_run_metrics

    #Function that deletes the model present in the manager
    def delete_model(self, model_name):
        if model_name in self.models:
            del self.models[model_name]
            self.updateGraphs()
            print(f"Deleted model: {model_name}")
        else:
            print(f"Model {model_name} not found.")

def __str__(self):
    return str(self.models)
