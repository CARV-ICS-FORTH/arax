package VineTalkInterface;
import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Platform;
import com.sun.jna.ptr.PointerByReference;
import com.sun.jna.Pointer;

public interface VineTalkInterface extends Library
{
	VineTalkInterface INSTANCE = (VineTalkInterface)Native.loadLibrary("libvine.so",VineTalkInterface.class);

	void vine_talk_init();
	int vine_accel_list(int type, boolean physical, PointerByReference accels);
	Pointer vine_accel_acquire_type (int type);
	int vine_accel_acquire_phys(PointerByReference accel);
	int vine_accel_release(PointerByReference accel);
	Pointer vine_proc_get(int type,String func_name);
	int vine_proc_put(Pointer proc);
	void vine_talk_exit();
}
