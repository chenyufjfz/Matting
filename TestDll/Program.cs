using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace TestDll
{
        // 摘要: 
    //     Representation of RGBA colors.
    public struct Color32
    {
        // 摘要: 
        //     Alpha component of the color.
        public byte a;
        //
        // 摘要: 
        //     Blue component of the color.
        public byte b;
        //
        // 摘要: 
        //     Green component of the color.
        public byte g;
        //
        // 摘要: 
        //     Red component of the color.
        public byte r;

        public Color32(byte r_, byte g_, byte b_, byte a_)
        {
            r = r_;
            g = g_;
            b = b_;
            a = a_;
        }

        public override string ToString()
        {
            string s;
            s = "r:" + r + " g:" + g + " b:" + b + " a:" + a;
            return s;
        }
    }

    class Program
    {
        [DllImport ("MattingDll")]
        static extern int MattingdllVersion();

        [DllImport("MattingDll")]
        static extern int MattingReset(int gpu, int width, int height);

        [DllImport("MattingDll")]
        static extern int MattingGetOutput(ref Color32 tex);

        [DllImport("MattingDll")]
        static extern int MattingGetLost();

        [DllImport("MattingDll")]
        static extern int MattingGetNoUpdate();

        [DllImport("MattingDll")]
        static extern void MattingStartProduce(int mode, int produce_rate);

        [DllImport("MattingDll")]
        static extern void MattingWaitProduceEnd(int mode);

        [DllImport("MattingDll")]
        static extern void MattingWriteDisk(ref Color32 tex, int wide, int height);

        public const int WIDE = 512;
        public const int HEIGHT = 384;

        static void Main(string[] args)
        {
            Color32[] tex = new Color32[WIDE * HEIGHT];
            MattingReset(0, WIDE, HEIGHT);
            MattingStartDiskProduce(1, 40);
            int count = 0;
	        while (true) {
		        if (MattingGetOutput(ref tex[0])==1) {
			        MattingWriteDisk(ref tex[0], WIDE, HEIGHT);
			        count = 0;
		        }
		        else
			        count++;		
		        if (count > 200)
			        break;
                Thread.Sleep(20);
	        }

	        MattingWaitProduceEnd(1);
        }
    }
}
