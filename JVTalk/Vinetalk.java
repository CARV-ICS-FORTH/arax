package Vinetalk;

import com.sun.jna.Native;
import com.sun.jna.NativeLibrary;
import com.sun.jna.Pointer;
import com.sun.jna.Memory;
import com.sun.jna.ptr.PointerByReference;
import java.io.Serializable;
import java.util.*;

public class Vinetalk implements Serializable
{
	public Vinetalk()
	{
		init();
	}
	
	public void init()
	{
		NativeLibrary.getInstance("rt");
		VineTalkInterface.INSTANCE.vine_talk_init();
	}

	public static VineProcedure acquireProcedure(VineAccelerator.Type type,String name)
	{
		Pointer proc;

		proc = VineTalkInterface.INSTANCE.vine_proc_get(type.getAsInt(),name);

		if(proc == Pointer.NULL)
			return null;
			
		return new VineProcedure(proc);
	}

	public static  VineAccelerator[] listAccelerators(VineAccelerator.Type type,Boolean physical)
	{
		PointerByReference ptr_ref = new PointerByReference();
		int accels = VineTalkInterface.INSTANCE.vine_accel_list(type.getAsInt(),physical,ptr_ref);
		System.out.println("Found "+accels+" accelerators");
		VineAccelerator [] accel_ar = new VineAccelerator[accels];
		int i = 0;
		for( Pointer ptr : ptr_ref.getValue().getPointerArray(0,accels) )
		{
			accel_ar[i++] = new VineAccelerator(ptr);
		}
		// Free ptr_ref
		Native.free(Pointer.nativeValue(ptr_ref.getValue())); // Not sure if this actually works...
		return accel_ar;
	}

	public static VineAccelerator acquireAccelerator(VineAccelerator.Type type)
	{
		return new VineAccelerator(VineTalkInterface.INSTANCE.vine_accel_acquire_type(type.getAsInt()));
	}

	public static VineAccelerator acquireAccelerator (VineAccelerator accel)
	{
		PointerByReference ptr_ref = new PointerByReference(accel.getPointer());
		VineTalkInterface.INSTANCE.vine_accel_acquire_phys(ptr_ref);
		return new VineAccelerator(ptr_ref.getValue());
	}

	public void exit()
	{
		VineTalkInterface.INSTANCE.vine_talk_exit();
	}
}
