import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Platform;
import Vinetalk.*;
import com.sun.jna.Pointer;



public class hello
{
	public static void main(String [] args)
	{
		Vinetalk vt = new Vinetalk();
		VineAccelerator acc = vt.listAccelerators(3,true)[0];
		System.out.println("Accelerator: "+acc);
		VineProcedure dg = vt.acquireProcedure(3,"darkGray");
		System.out.println("DarkGray: "+dg);
		DarkGrayArgs dka = new DarkGrayArgs();
		dka.width = 1024;
		dka.height = 1024;
		VineTask task = new VineTask(dg,dka);

		acc.issue(task);

		acc.release();
		dg.release();
		vt.exit();
	}
}
