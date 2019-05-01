using MulticlassClassification.DataStructures;
using MulticlassClassification.Repositories;
using System;
using System.Collections.Generic;
using System.Text;

namespace MulticlassClassification
{
    public class IrisDataPrediction
    {
        public IEnumerable<string> Categories { get; set; }

        private readonly IIrisDataRepository irisDataRepository;
        public IrisDataPrediction(IIrisDataRepository irisDataRepository)
        {
            this.irisDataRepository = irisDataRepository;
        }

        public IEnumerable<IrisData> Data()
        {
            var dataParser = new IrisDataParser();
            return irisDataRepository.GetIrisData(dataParser.RelativeFilePath);
        }

        public IEnumerable<Dictionary<string, float>> Probabilities()
        {
            return irisDataRepository.GetProbabilities();
        }
    }
}
