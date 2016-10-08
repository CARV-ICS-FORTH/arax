package Vinetalk;
import VineTalkInterface.*;
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
		VineTalkInterface.INSTANCE.vine_task_issue(getPointer(),task.getProcedure(),task.getArgs().getPointer(),0,null,0,null);
	}

	public void release()
	{
		PointerByReference ptr_ref = new PointerByReference(getPointer());
		VineTalkInterface.INSTANCE.vine_accel_release(ptr_ref);
	}
}
