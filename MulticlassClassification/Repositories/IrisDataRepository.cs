using MulticlassClassification.DataStructures;
using System;
using System.Collections.Generic;
using System.Text;

namespace MulticlassClassification.Repositories
{
    public interface IIrisDataRepository
    {
        IEnumerable<IrisData> GetIrisData(string textfilePath);
        IEnumerable<Dictionary<string, float>> GetProbabilities();
    }
    public class IrisDataRepository : IIrisDataRepository
    {
        private readonly IIrisDataParser irisDataParser;
        public IrisDataRepository(IIrisDataParser irisDataParser)
        {
            this.irisDataParser = irisDataParser;
        }

        public IEnumerable<IrisData> GetIrisData(string textFilePath)
        {
            var irisData = irisDataParser.Parse(textFilePath);
            return irisData;
        }

        //TODO: Make this actually do something. Reconsider how Program.PredictValues() is designed
        public IEnumerable<Dictionary<string, float>> GetProbabilities()
        {
            var probs = new List<Dictionary<string, float>>();
            //var probs = ClassificationModel.PredictValues();
            return probs;
        }

    }
}
