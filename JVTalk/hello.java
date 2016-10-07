import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.NativeLibrary;
import com.sun.jna.Platform;
import VineTalkInterface.*;

public class hello
{
	public static void main(String [] args)
	{
		NativeLibrary.getInstance("rt");
		VineTalkInterface test = VineTalkInterface.INSTANCE;
		test.vine_talk_init();
		System.out.println("Hello World");
		test.vine_talk_exit();
	}
}
