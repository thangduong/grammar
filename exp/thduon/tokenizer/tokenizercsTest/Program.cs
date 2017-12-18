using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TokenizerCS;

namespace tokenizercsTest
{
    class Program
    {
        static void Main(string[] args)
        {
            Tokenizer tok = new Tokenizer();
            var result = tok.Tokenize("F.D.R. was a great president in the 1930s.");
            foreach (var token in result)
            {
                System.Console.WriteLine(token.token);
            }
        }
    }
}
