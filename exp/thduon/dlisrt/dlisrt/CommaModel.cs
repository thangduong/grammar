using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TensorFlow;
using Newtonsoft.Json;

namespace OXOGrammarModels
{
    public class CommaModel : DeepLearning.IDLWindowsModel, IDisposable
    {
        Dictionary<string, int> _vocab;
        TFGraph _graph;
        TFSession _session;
        string _prefix = "commaV5";
        private bool _disposed = false;

        public override void Initialize()
        {
            string vocab_json = File.ReadAllText("vocab.json");
            _vocab = JsonConvert.DeserializeObject<Dictionary<string, int>>(vocab_json);
            _graph = new TFGraph();
            TFImportGraphDefOptions opts = new TFImportGraphDefOptions();
            TFBuffer buff = new TFBuffer(File.ReadAllBytes("commaV5.graphdef"));
            opts.SetPrefix(_prefix);
            _graph.Import(buff, opts);
            _session = new TFSession(_graph);
        }

        public int[] IndexText(string strSentence)
        {  
            string[] sentencePieces = strSentence.Split(' ');
            int[] indexedPieces = new int[sentencePieces.Length];
            for (var i = 0; i < sentencePieces.Length; i++)
            {
                if (_vocab.ContainsKey(sentencePieces[i]))
                    indexedPieces[i] = _vocab[sentencePieces[i]];
                else
                    indexedPieces[i] = _vocab["unk"];
            }
            return indexedPieces;
        }

        public override string Execute(string input)
        {
            var inputdict = JsonConvert.DeserializeObject<Dictionary<string, string>>(input);
            float conf = -1.0f;
            string strSentence = inputdict["sentence"];
            
            // for now, just tokenize like this!
            var sentence_node = _graph[_prefix + "/" + "sentence"];
            var is_training_node = _graph[_prefix + "/" + "is_training"];
            var sm_decision_node = _graph[_prefix + "/" + "sm_decision"];
            
            var runner = _session.GetRunner();
            int[][] sentenceTensor = new int[1][];
            sentenceTensor[0] = IndexText(strSentence);
            TFTensor sentence = new TFTensor(sentenceTensor);
            
            runner.AddInput(sentence_node[0], sentence);
            runner.AddInput(is_training_node[0], new TFTensor(false));
            runner.Fetch(sm_decision_node[0]);
            
            var output = runner.Run();
            
            // Fetch the results from output:
            TFTensor result = output[0];
            object o = result.GetValue();
            conf = ((float[,])o)[0,1];
            return string.Format("{{ conf: {0:N2} }}", conf);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    _session.Dispose();
                    _graph.Dispose();
                }
                _disposed = true;
            }
        }
    }
}
