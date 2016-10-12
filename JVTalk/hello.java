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
		System.out.println("Accelerator: "+acc);
		VineProcedure dg = vt.acquireProcedure(3,"darkGray");
		System.out.println("DarkGray: "+dg);
		System.out.println("Input : "+args[0]);
		System.out.println("Output: "+args[1]);
		VineTask task = new VineTask(dg);
		byte [] output = null;
		try
		{
			BufferedImage image = ImageIO.read(new File(args[0]));
			DataBufferByte dbb = (DataBufferByte)image.getRaster().getDataBuffer();
			DarkGrayArgs dka = new DarkGrayArgs();
			dka.width = image.getWidth();
			dka.height = image.getHeight();
			byte [] data = dbb.getData();
			output = new byte[data.length/3]; // Output is 1 channe
			System.out.println("Image size:"+dka);
			task.setArgs(dka);
			task.addInput(data);
			task.addOutput(output);

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
