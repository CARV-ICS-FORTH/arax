package Vinetalk;

public static interface VineDataFlags
{
	public static final int NONE_SYNC  = 0;
	public static final int USER_SYNC = 1;
	public static final int SHM_SYNC = 2;
	public static final int REMT_SYNC = 4;
	public static final int ALL_SYNC  = 7;
	public static final int FREE = 8;
}
