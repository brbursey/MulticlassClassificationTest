using System.Collections.Generic;
using System.IO;
using System.Linq;
using MulticlassClassification.Iris.DataStructures;

namespace MulticlassClassification.Iris
{
    public interface IIrisDataParser
    {
        IEnumerable<IrisData> Parse(string filePath);
    }

    public class IrisDataParser : IIrisDataParser
    {
        public static string FileName = "FullData.txt";
        public string RelativeFilePath = $".\\Data\\{FileName}";
      
        public IEnumerable<IrisData> Parse(string filePath)
        {
            var lines = File.ReadLines(filePath);
            var objectList = new List<IrisData>();
            foreach (var line in lines)
            {
                var features = line.Split('\t').Select(float.Parse).ToList();
                objectList.Add(new IrisData()
                {
                    SepalLength = features[0],
                    SepalWidth = features[1],
                    PetalLength = features[2],
                    PetalWidth = features[3],
                }
                );
            }
            return objectList;
        }
    }
}
