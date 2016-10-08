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

	public void issue()
	{
	}

	public void release()
	{
		PointerByReference ptr_ref = new PointerByReference(getPointer());
		VineTalkInterface.INSTANCE.vine_accel_release(ptr_ref);
	}
}
