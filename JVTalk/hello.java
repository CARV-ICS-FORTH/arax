import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Platform;
import Vinetalk.*;
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

public class hello
{
	public static void main(String [] args)
	{
		Vinetalk vt = new Vinetalk();
		VineAccelerator acc = vt.listAccelerators(VineAccelerator.Type.CPU,true)[0];

		if(args.length != 1)
		{
			System.out.println("Usage:\njava -jar <this jar> <input_string>");
			return;
		}
		System.out.println("Accelerator: "+acc);
		VineProcedure dg = vt.acquireProcedure(VineAccelerator.Type.CPU,"noop");
		System.out.println("Noop: "+dg);
		System.out.println("Input : \""+args[0] + "\"size: "+ args[0].length());
		VineTask task = new VineTask(dg);
		byte [] output = null;
		byte [] input = null;
		try
		{
			input =  Arrays.copyOf(args[0].getBytes(),args[0].length()+1);
			output = new byte[args[0].length()+1];
//			float [] test = new float[10];
//			task.addInput(input);
			task.addInput(input);
			task.addInput(input);
//			task.addOutput(output);
			System.out.println("Press <any> key");
			System.in.read();
		}catch(IOException e)
		{
			e.printStackTrace();
		}
		acc.issue(task);
		System.out.println("Status: "+task.status());
		System.out.println("Got \'"+new String(output)+"\' back!");
		dg.release();
		vt.exit();
	}
}
