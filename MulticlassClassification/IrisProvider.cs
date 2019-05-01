using System.IO;
using System.Reflection.Emit;
using Microsoft.ML;
using MulticlassClassification.DataStructures;

namespace MulticlassClassification
{
    public interface IProvider
    {
        string BaseModelPath { get; set; }
        string ModelZipFilePath { get; set; }
        string ModelPath { get; set; }
        IDataView TrainingDataView { get; set; }
        IDataView TestDataView { get; set; }
        IModelBuilder ModelBuilder { get; set; }
        
    }
    public class IrisProvider : IProvider
    {
        public string BaseModelPath { get; set; }
        public string ModelZipFilePath { get; set; }
        public string ModelPath { get; set; }
        public string TrainDataPath { get; set; }
        public string TestDataPath { get; set; }
        public IDataView TrainingDataView { get; set; }
        public IDataView TestDataView { get; set; }
        public IModelBuilder ModelBuilder { get; set; }

        public IrisProvider(MLContext context)
        {
            BaseModelPath = ".\\MLModels";
            ModelZipFilePath = $"{BaseModelPath}\\TestClassificationModel.zip";
            ModelPath = GetAbsolutePath($"{BaseModelPath}\\TestClassificationModel.zip");
            TrainDataPath = GetAbsolutePath(".\\Data\\TrainData.txt");
            TestDataPath = GetAbsolutePath(".\\Data\\TestData.txt");
            TrainingDataView = context.Data.LoadFromTextFile<IrisData>(TrainDataPath, hasHeader: true);
            TestDataView = context.Data.LoadFromTextFile<IrisData>(TestDataPath, hasHeader: true);
            ModelBuilder = new IrisModelBuilder();
        }

        public static string GetAbsolutePath(string relativePath)
        {
            var dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }
    }
}