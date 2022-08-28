package Arax;
import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Platform;
import com.sun.jna.ptr.PointerByReference;
import com.sun.jna.Pointer;
import com.sun.jna.Structure;

public interface AraxInterface extends Library
{
	AraxInterface INSTANCE = (AraxInterface)Native.loadLibrary("arax",AraxInterface.class);

	void arax_init();
	int arax_accel_list(int type, boolean physical, PointerByReference accels);
	Pointer arax_accel_acquire_type (int type);
	int arax_accel_acquire_phys(PointerByReference accel);
	int arax_accel_release(PointerByReference accel);
	int arax_vaccel_queue_size(Pointer vaccel);
	Pointer arax_proc_get(String func_name);
	Pointer arax_task_issue (Pointer accel, Pointer proc, Pointer args, long args_size, long in_count, Pointer[] input, long out_count, Pointer[] output);
	int arax_task_wait (Pointer task);
	int arax_task_stat (Pointer task,Pointer stat);
	int arax_task_free (Pointer task);
	void arax_data_get(Pointer data, Pointer user);
	void arax_data_set(Pointer data, Pointer accel, Pointer user);
	void arax_data_free(Pointer data);
	int arax_proc_put(Pointer proc);
	void arax_exit();
	Pointer ARAX_BUFFER(long size);
}
