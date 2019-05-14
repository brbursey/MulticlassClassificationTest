using System.Collections.Generic;
using MulticlassClassification.Iris.DataStructures;

namespace MulticlassClassification.Iris.Repositories
{
    public interface IIrisDataRepository
    {
        IEnumerable<IrisData> GetIrisData(string textfilePath);
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
    }
}
