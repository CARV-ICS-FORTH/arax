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

public class hello
{
	public static void main(String [] args)
	{
		Vinetalk vt = new Vinetalk();
		VineAccelerator acc = vt.listAccelerators(3,true)[0];

		if(args.length != 1)
		{
			System.out.println("Usage:\njava -jar <this jar> <input_string>");
			return;
		}

		System.out.println("Accelerator: "+acc);
		VineProcedure dg = vt.acquireProcedure(3,"noop");
		System.out.println("Noop: "+dg);
		System.out.println("Input : \""+args[0] + "\"size: "+ args[0].length());
		VineTask task = new VineTask(dg);
		byte [] output = null;
		byte [] input = null;
		try
		{
			input = args[0].getBytes();
			output = new byte[args[0].length()+1];
			task.addInput(input);
			task.addOutput(output);
			System.out.println("Press <any> key");
			System.in.read();
		}catch(IOException e)
		{
			e.printStackTrace();
		}
		acc.issue(task);
		System.out.println("Status: "+task.status());
		try {
			new FileOutputStream(new File(args[1])).write(output);
		} catch (IOException e) {
			e.printStackTrace();
		}		acc.release();
		dg.release();
		vt.exit();
	}
}
