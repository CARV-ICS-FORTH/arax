package Vinetalk;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;

public class VineAccelerator extends VineObject
{
	public VineAccelerator(Pointer ptr)
	{
		super(ptr);
	}

	public void issue(VineTask task)
	{
		VineBuffer[] in = task.getInputs();
		VineBuffer[] out = task.getOutputs();
		Pointer args = (task.getArgs()!=null)?task.getArgs().getPointer():null;
		task.setTask(VineTalkInterface.INSTANCE.vine_task_issue(getPointer(),task.getProcedure(),args,in.length,in,out.length,out));
	}

	public void release()
	{
		PointerByReference ptr_ref = new PointerByReference(getPointer());
		VineTalkInterface.INSTANCE.vine_accel_release(ptr_ref);
	}
}
