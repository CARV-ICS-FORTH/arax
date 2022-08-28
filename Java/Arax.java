package Arax;

import com.sun.jna.Native;
import com.sun.jna.NativeLibrary;
import com.sun.jna.Pointer;
import com.sun.jna.Memory;
import com.sun.jna.ptr.PointerByReference;
import java.io.Serializable;
import java.util.*;

public class Arax implements Serializable
{
	public Arax()
	{
		init();
	}

	public void init()
	{
		NativeLibrary.getInstance("rt");
		AraxInterface.INSTANCE.arax_init();
	}

	public static AraxProcedure acquireProcedure(AraxAccelerator.Type type,String name)
	{
		Pointer proc;

		proc = AraxInterface.INSTANCE.arax_proc_get(name);

		if(proc == Pointer.NULL)
			throw new RuntimeException("acquireProcedure("+type+",'"+name+"'): Kernel not found");

		return new AraxProcedure(proc);
	}

	public static  AraxAccelerator[] listAccelerators(AraxAccelerator.Type type,Boolean physical)
	{
		PointerByReference ptr_ref = new PointerByReference();
		int accels = AraxInterface.INSTANCE.arax_accel_list(type.getAsInt(),physical,ptr_ref);
		System.out.println("Found "+accels+" accelerators");
		AraxAccelerator [] accel_ar = new AraxAccelerator[accels];
		int i = 0;
		for( Pointer ptr : ptr_ref.getValue().getPointerArray(0,accels) )
		{
			accel_ar[i++] = new AraxAccelerator(ptr);
		}
		// Free ptr_ref
		Native.free(Pointer.nativeValue(ptr_ref.getValue())); // Not sure if this actually works...
		return accel_ar;
	}

	public static AraxAccelerator acquireAccelerator(AraxAccelerator.Type type)
	{
		return new AraxAccelerator(AraxInterface.INSTANCE.arax_accel_acquire_type(type.getAsInt()));
	}

	public static AraxAccelerator acquireAccelerator (AraxAccelerator accel)
	{
		PointerByReference ptr_ref = new PointerByReference(accel.getPointer());
		AraxInterface.INSTANCE.arax_accel_acquire_phys(ptr_ref);
		return new AraxAccelerator(ptr_ref.getValue());
	}

	public void exit()
	{
		AraxInterface.INSTANCE.arax_exit();
	}
}
