package Vinetalk;
import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Platform;
import com.sun.jna.ptr.PointerByReference;
import com.sun.jna.Pointer;
import com.sun.jna.Structure;

public interface VineTalkInterface extends Library
{
	VineTalkInterface INSTANCE = (VineTalkInterface)Native.loadLibrary("vine",VineTalkInterface.class);

	void vine_talk_init();
	int vine_accel_list(int type, boolean physical, PointerByReference accels);
	Pointer vine_accel_acquire_type (int type);
	int vine_accel_acquire_phys(PointerByReference accel);
	int vine_accel_release(PointerByReference accel);
	int vine_vaccel_queue_size(Pointer vaccel);
	Pointer vine_proc_get(int type,String func_name);
	Pointer vine_task_issue (Pointer accel, Pointer proc, Pointer args, long args_size, long in_count, Pointer[] input, long out_count, Pointer[] output);
	int vine_task_wait (Pointer task);
	int vine_task_stat (Pointer task,Pointer stat);
	int vine_task_free (Pointer task);
	void vine_data_sync_to_remote(Pointer accel,Pointer data,int block);
	void vine_data_sync_from_remote(Pointer accel,Pointer data,int block);
	void vine_data_modified(Pointer data,int where);
	void vine_data_set_arch(Pointer data,int arch);
	int vine_proc_put(Pointer proc);
	void vine_talk_exit();
	Pointer VINE_BUFFER(Pointer user_buffer,long size);
}
