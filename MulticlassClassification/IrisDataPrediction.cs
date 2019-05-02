using MulticlassClassification.DataStructures;
using MulticlassClassification.Repositories;
using System.Collections.Generic;

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

        private IEnumerable<IrisData> GetData()
        {
            var dataParser = new IrisDataParser();
            return irisDataRepository.GetIrisData(dataParser.RelativeFilePath);
        }
    }
}
                                