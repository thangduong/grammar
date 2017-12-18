using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Runtime.InteropServices;

namespace TokenizerCS
{
    public class Tokenizer : IDisposable
    {
        [DllImport(@"tokenizer.dll")]
        public static extern IntPtr LoadTokenizer();
        [DllImport(@"tokenizer.dll")]
        public static extern void UnloadTokenizer(IntPtr tokenizer);
        [DllImport(@"tokenizer.dll")]
        public static extern IntPtr TokenizeString(IntPtr tokenizer, string input_string, bool translit);
        [DllImport(@"tokenizer.dll")]
        public static extern IntPtr NextToken(IntPtr result, ref int start, ref int len, ref int type);
        [DllImport(@"tokenizer.dll")]
        public static extern void FreeTokenizedResult(IntPtr result);

        IntPtr _tokenizer;

        /// <summary>
        /// Token structure contains the string token, the start position
        /// of the token in the larger string, the length, and the type (for regex).
        /// </summary>
        public struct Token {
            public string token;
            public int start, len, type;
        };

        public Tokenizer()
        {
            _tokenizer = LoadTokenizer();
        }

        /// <summary>
        /// Tokenize a string using the C++ native tokenizer
        /// </summary>
        /// <param name="input_string">Input string to tokenize.  Try "F.D.R. was a great president in the 1930s."</param>
        /// <param name="translit">True to turn on UTF-8 to ASCII translit</param>
        /// <returns>A list of tokens</returns>
        public List<Token> Tokenize(string input_string, bool translit=true)
        {
            List<Token> token_list = new List<Token>();
            IntPtr tokenized_result = TokenizeString(_tokenizer, input_string, translit);
            Token token;
            token.start = token.len = token.type = 0;
            IntPtr native_string = IntPtr.Zero;
            while ((native_string = NextToken(tokenized_result, ref token.start, ref token.len, ref token.type)) != IntPtr.Zero) {
                token.token = Marshal.PtrToStringAnsi(native_string);
                token_list.Add(token);
            }
            return token_list;
        }



        bool _disposed = false;

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (_disposed)
                return;

            if (_disposed)
            {
                UnloadTokenizer(_tokenizer);
            }

            // Free any unmanaged objects here.
            //
            _disposed = true;
        }
    }
}
