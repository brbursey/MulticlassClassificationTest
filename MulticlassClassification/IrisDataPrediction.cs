using MulticlassClassification.DataStructures;
using MulticlassClassification.Repositories;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Emit;

namespace MulticlassClassification
{
    public class IrisDataPrediction
    {
        public readonly IEnumerable<string> Categories = new List<string>
        {
            "Setosa",
            "Virginica",
            "Versicolor"
        };
        public IEnumerable<IrisData> Data { get; }
        public IEnumerable<Dictionary<string, float>> Probabilities { get; set; }

        private readonly IIrisDataRepository irisDataRepository;
        public IrisDataPrediction(IIrisDataRepository irisDataRepository)
        {
            this.irisDataRepository = irisDataRepository;
            Data = GetData();
        }

        public IEnumerable<string> PredictCategory(IEnumerable<Dictionary<string, float>> probabilities)
        {
            var prediction = new List<string>();
            foreach (var sample in probabilities)
            {
                var max = 0f;
                var label = "";
                foreach (var category in sample)
                {
                    if (category.Value >= max)
                    {
                        max = category.Value;
                        label = category.Key;
                    }
                }
                //fix this line
                prediction.Add(label);
            }
            return prediction;
        }

        public IEnumerable<Iris> PredictedData(IEnumerable<IrisData> inputData, IEnumerable<string> predictedCategories)
        {
            var irises = inputData.Zip(predictedCategories, (data, category) => new Iris()
            {
                Label = category,
                SepalLength = data.SepalLength,
                SepalWidth = data.SepalWidth,
                PetalLength = data.PetalLength,
                PetalWidth = data.PetalWidth
            });
            return irises;
        }

        private IEnumerable<IrisData> GetData()
        {
            var dataParser = new IrisDataParser();
            return irisDataRepository.GetIrisData(dataParser.RelativeFilePath);
        }
    }
}
                                