package Vinetalk;
import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Platform;
import com.sun.jna.ptr.PointerByReference;
import com.sun.jna.Pointer;
import com.sun.jna.Structure;


public interface VineTalkInterface extends Library
{
	VineTalkInterface INSTANCE = (VineTalkInterface)Native.loadLibrary("libvine.so",VineTalkInterface.class);

	void vine_talk_init();
	int vine_accel_list(int type, boolean physical, PointerByReference accels);
	Pointer vine_accel_acquire_type (int type);
	int vine_accel_acquire_phys(PointerByReference accel);
	int vine_accel_release(PointerByReference accel);
	Pointer vine_proc_get(int type,String func_name);
	Pointer vine_task_issue (Pointer accel, Pointer proc, Pointer args, long in_count, Structure[] input, long out_count, Structure[] output);
	int vine_task_wait (Pointer task);
	int vine_task_stat (Pointer task,Pointer stat);
	int vine_task_free (Pointer task);
	int vine_proc_put(Pointer proc);
	void vine_talk_exit();
}
