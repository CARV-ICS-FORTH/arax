package Vinetalk;
import com.sun.jna.Pointer;

public class VineProcedure extends VineObject
{
	public VineProcedure(Pointer ptr)
	{
		super(ptr);
	}
	public void release()
	{
		VineTalkInterface.INSTANCE.vine_proc_put(getPointer());
	}
}
