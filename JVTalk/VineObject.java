package Vinetalk;
import com.sun.jna.Pointer;
import com.sun.jna.Structure;
import java.util.List;
import java.util.Arrays;

public abstract class VineObject
{
	public class cRep extends Structure
	{
		public Pointer prev;
		public Pointer next;
		public Pointer owner;
		/* TODO:Probably should make the above a seperate struct */
		public int type;
		public byte[] name = new byte[32];	// Not sure if 'proper'
		public cRep(Pointer ptr)
		{
			super(ptr);
			read();
		}
		protected List<String> getFieldOrder()
		{
			return Arrays.asList(new String[] { "prev", "next", "owner","type","name"});
		}
	}

	public VineObject(Pointer ptr)
	{
		this.ptr = ptr;
		crep = new cRep(ptr);
	}

	public String getName()
	{
		return new String(crep.name);
	}

	public Pointer getPointer()
	{
		return ptr;
	}

	public void setPointer(Pointer ptr)
	{
		this.ptr = ptr;
	}

	public String toString()
	{
		return this.getClass().getName()+"("+new String(crep.name)+","+crep.type+")@0x"+Long.toHexString(Pointer.nativeValue(crep.owner));
	}
	public abstract void release();
	private Pointer ptr;
	private cRep crep;
}
