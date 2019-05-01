using Microsoft.ML;

namespace MulticlassClassification
{
    public class ModelPathInfo
    {
        public string ModelPath;
        public string ModelZipFilePath;
        public IDataView TrainingDataView;
        public IDataView TestDataView;
    }
}