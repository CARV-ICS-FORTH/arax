package Arax;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;

public class AraxAccelerator extends AraxObject
{
	public enum Type {
		ANY(0),       /**< Let Scheduler Decide */
		GPU(1),       /**< Run on GPU with CUDA */
		GPU_SOFT(2),  /**< Run on CPU with software CUDA(Useful for debug?) */
		CPU(3),       /**< Run Native x86 code */
		FPGA(4),      /**< Custom Fpga accelerator */
		NANO_ARM(5),  /**< ARM accelerator core from NanoStream */
		NANO_CORE(6), /**< NanoStreams FPGA accelerator */
		ARAX_ACCEL_TYPES(7); /** End Marker */
		private final int value;

		Type(int value)
		{this.value = value;}
		int getAsInt()
		{return value;}
	}

	public AraxAccelerator(Pointer ptr)
	{
		super(ptr);
	}

	public int getQueueSize()
	{
		int ret = AraxInterface.INSTANCE.arax_vaccel_queue_size(getPointer());
		if(ret == -1)
			throw new RuntimeException("Could not get queue size!");
		return ret;
	}

	public void issue(AraxTask task)
	{
		Pointer[] in = task.getInputs();
		int in_len = (in!=null)?in.length:0;
		Pointer[] out = task.getOutputs();
		int out_len = (out!=null)?out.length:0;
		Pointer proc = task.getProcedure();
		if(proc == null)
			throw new RuntimeException("Issuing task with NULL procedure");
		task.setTask(AraxInterface.INSTANCE.arax_task_issue(getPointer(),proc,task.getArg(),task.getArgSize(),in_len,in,out_len,out));
	}

	public void release()
	{
		PointerByReference ptr_ref = new PointerByReference(getPointer());
		AraxInterface.INSTANCE.arax_accel_release(ptr_ref);
	}
}
