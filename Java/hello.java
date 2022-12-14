import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Platform;
import Arax.*;
import com.sun.jna.Pointer;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import javax.imageio.ImageIO;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayInputStream;
import java.io.FileOutputStream;
import java.util.Arrays;
import com.sun.jna.Structure;
import java.util.List;
public class hello
{
	public static class Args extends Structure
	{
		public int magic;
		public Args(int magic)
		{
			this.magic = magic;
		}
		protected List<String> getFieldOrder()
		{
			return Arrays.asList(new String[] {"magic"});
		}
	}

	public static void main(String [] args)
	{
		if(args.length != 1)
		{
			System.out.println("Usage:\njava -jar <this jar> <input_string>");
			return;
		}

		Arax vt = new Arax();
		AraxAccelerator acc = vt.acquireAccelerator(AraxAccelerator.Type.CPU);

		System.out.println("Accelerator: "+acc);
		AraxProcedure dg = vt.acquireProcedure(AraxAccelerator.Type.CPU,"noop");
		System.out.println("Noop: "+dg);
		System.out.println("Input : \""+args[0] + "\"size: "+ args[0].length());
		AraxTask task = new AraxTask(dg);
		byte [] output = null;
		byte [] input = null;
		AraxBuffer in = null;
		AraxBuffer out= null;
		try
		{
			input =  Arrays.copyOf(args[0].getBytes(),args[0].length()+1);
			output = new byte[args[0].length()+1];
			Args nargs = new Args(1337);
			in = new AraxBuffer(input.length);
			in.set(acc,input);
			out = new AraxBuffer(output.length);
			task.setArgs(nargs)
				.addInput(in)
				.addOutput(out);
			System.out.println("Press <any> key");
			System.in.read();
		}catch(IOException e)
		{
			e.printStackTrace();
		}
		acc.issue(task);
		System.out.println("Status: "+task.status());
		out.get(output);
		System.out.println("Got \'"+new String(output)+"\' back!");
		in.free();
		out.free();
		dg.release();
		acc.release();
		vt.exit();
	}
}
