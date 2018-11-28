using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;


[System.Serializable]
public class TestJointMuscle
{
    public string Name;
    public TestJointBodyHelper.MuscleGroup Group;


    [Range(-1,1)]
    public float TargetAngularVelocityX;
    // [Range(-1,1)]
    // public float TargetAngularVelocityY;
    // [Range(-1,1)]
    // public float TargetAngularVelocityZ;

    public Vector3 MaximumForce;
    [Range(-1,1)]

    public float TargetNormalizedAngleX;
    public Vector3 ObsLocalByAngle;
    public float ObsAngleMagnitude;
    public float ObsAngle;
    public float ObsRotationVelocity;

    [Range(-1,1)]
    public float ObsNormalizedAngleX;


    // public Rigidbody Rigidbody;
    public Transform Transform;
    public ConfigurableJoint ConfigurableJoint;
    // public Rigidbody Parent;
    float? _lastObseAngleNormalized;

    // public Quaternion DefaultLocalRotation;
    // public Quaternion ToJointSpaceInverse;
    // public Quaternion ToJointSpaceDefault;
    // public Quaternion InitialRootRotation;
    // public Vector3 InitialRootPosition;
    // public ConfigurableJoint RootConfigurableJoint;




    bool _hasRanVeryFirstInit;
    public void Init()
    {
        // Rigidbody.angularVelocity = Vector3.zero;
        // Rigidbody.velocity = Vector3.zero;
        _lastObseAngleNormalized = null;

        if (!_hasRanVeryFirstInit) {
			// Parent = ConfigurableJoint.connectedBody;
            // InitialRootRotation = RootConfigurableJoint.transform.rotation;
            // InitialRootPosition = RootConfigurableJoint.transform.position;
            // DefaultLocalRotation = LocalRotation;
            // Vector3 forward = this.Transform.forward;
            // Vector3 up = this.Transform.forward;
			// Quaternion toJointSpace = Quaternion.LookRotation(forward, up);
			
			// ToJointSpaceInverse = Quaternion.Inverse(toJointSpace);
			// ToJointSpaceDefault = DefaultLocalRotation * toJointSpace;
            _hasRanVeryFirstInit = true;
        }
    }

    public void UpdateObservations()
    {
        ObsLocalByAngle = Vector3.Scale(Transform.localRotation.eulerAngles, ConfigurableJoint.axis);
        ObsAngleMagnitude = ObsLocalByAngle.magnitude;
        ObsAngle = MagnitudeToAngle(ObsAngleMagnitude);

        var obsAngleNormalized = ObsAngle/180f;
        ObsRotationVelocity = 0f;
        if (_lastObseAngleNormalized.HasValue)
            ObsRotationVelocity = obsAngleNormalized - _lastObseAngleNormalized.Value;
        ObsRotationVelocity = ObsRotationVelocity / UnityEngine.Time.fixedDeltaTime;
        _lastObseAngleNormalized = obsAngleNormalized;
        float min = ConfigurableJoint.lowAngularXLimit.limit;
        float max = ConfigurableJoint.highAngularXLimit.limit;
        float spread = max-min;
        float mid = min + (spread/2f);
        float angle = Mathf.Clamp(ObsAngle, min, max);
        ObsNormalizedAngleX = (angle-mid) / (spread/2f);
    }
    float MagnitudeToAngle(float magnitude)
    {
        return magnitude <= 180f ? -magnitude : (360f-magnitude);
    }
    public float GetReward()
    {
        // if (ObsNormalizedAngleX < 0.01f && ObsNormalizedAngleX > -0.01f)
        //     return 1f;
        // return 0f;
        var diff = TargetNormalizedAngleX-ObsNormalizedAngleX;
        var reward = 1f-Mathf.Abs(diff);
        return reward;
    }

    public void UpdateMotor()
    {
		var t = ConfigurableJoint.targetAngularVelocity;
		t.x = TargetAngularVelocityX * MaximumForce.x;
		// t.y = TargetAngularVelocityY * MaximumForce.y;
		// t.z = TargetAngularVelocityZ * MaximumForce.z;
		ConfigurableJoint.targetAngularVelocity = t;

		var angX = ConfigurableJoint.angularXDrive;
		angX.positionSpring = 1f;
		var scale = MaximumForce.x * Mathf.Pow(Mathf.Abs(TargetAngularVelocityX), 3);
		angX.positionDamper = Mathf.Max(1f, scale);
		angX.maximumForce = Mathf.Max(1f, MaximumForce.x);
		ConfigurableJoint.angularXDrive = angX;

        // var maxForce = (MaximumForce.y + MaximumForce.z) / 2;
		// var angYZ = ConfigurableJoint.angularYZDrive;
		// angYZ.positionSpring = 1f;
		// scale = maxForce * Mathf.Pow((Mathf.Abs(TargetAngularVelocityY) + Mathf.Abs(TargetAngularVelocityZ))/2, 3);
		// angYZ.positionDamper = Mathf.Max(1f, scale);
		// angYZ.maximumForce = Mathf.Max(1f, maxForce);
		// ConfigurableJoint.angularYZDrive = angYZ;
	}    
    // static Vector3 NormalizedEulerAngles(Vector3 eulerAngles)
    // {
    //     var x = eulerAngles.x < 180f ?
    //         eulerAngles.x :
    //         - 360 + eulerAngles.x;
    //     var y = eulerAngles.y < 180f ?
    //         eulerAngles.y :
    //         - 360 + eulerAngles.y;
    //     var z = eulerAngles.z < 180f ?
    //         eulerAngles.z :
    //         - 360 + eulerAngles.z;
    //     x = x / 180f;
    //     y = y / 180f;
    //     z = z / 180f;
    //     return new Vector3(x,y,z);
    // }

    // public Quaternion LocalRotation {
    //     get {
    //         // around root Rotation 
    //         return Quaternion.Inverse(RootRotation) * Transform.rotation;

    //         // around parent space
    //         // return Quaternion.Inverse(ParentRotation) * transform.rotation;
    //     }
    // }

    // public Quaternion RootRotation{
    //     get {
    //         return InitialRootRotation;
    //     }
    // }

    // static Vector3 ScaleNormalizedByJoint(Vector3 normalizedRotation, ConfigurableJoint configurableJoint)
    // {
    //     var x = normalizedRotation.x > 0f ?
    //         (normalizedRotation.x * 180f) / configurableJoint.highAngularXLimit.limit :
    //         (-normalizedRotation.x * 180f) / configurableJoint.lowAngularXLimit.limit;
    //     var y = (normalizedRotation.y * 180f) / configurableJoint.angularYLimit.limit;
    //     var z = (normalizedRotation.z * 180f) / configurableJoint.angularZLimit.limit;
    //     var scaledNormalizedRotation = new Vector3(x,y,z);
    //     return scaledNormalizedRotation;
    // }


}