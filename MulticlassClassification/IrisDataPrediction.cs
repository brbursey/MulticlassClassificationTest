using MulticlassClassification.DataStructures;
using MulticlassClassification.Repositories;
using System;
using System.Collections.Generic;
using System.Text;

namespace MulticlassClassification
{
    public class IrisDataPrediction
    {
        private readonly IEnumerable<string> Categories = new List<string>
        {
            "Setosa",
            "Virginica",
            "Versicolor"
        };
        public IEnumerable<IrisData> Data { get; set; }

        private readonly IIrisDataRepository irisDataRepository;
        private readonly IProvider irisProvider;
        public IrisDataPrediction(IIrisDataRepository irisDataRepository, IProvider irisProvider)
        {
            this.irisDataRepository = irisDataRepository;
            this.irisProvider = irisProvider;
        }

        private IEnumerable<IrisData> GetData()
        {
            var dataParser = new IrisDataParser();
            return irisDataRepository.GetIrisData(dataParser.RelativeFilePath);
        }

        public IEnumerable<Dictionary<string, float>> Probabilities()
        {
            var irisData = GetData();
            var classificationModel = new ClassificationModel(irisProvider);
            var probabilities = classificationModel.PredictValues(irisData, Categories);
            return probabilities;
        }

    }
}
                                