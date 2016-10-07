package VineTalkInterface;
import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Platform;

public interface VineTalkInterface extends Library
{
	VineTalkInterface INSTANCE = (VineTalkInterface)Native.loadLibrary("libvine.so",VineTalkInterface.class);

	void vine_talk_init();
	void vine_talk_exit();
}
